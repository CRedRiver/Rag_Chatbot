import os
import json
import time
import random
import textwrap
from typing import Optional, Any

from google import genai
from google.genai import types

class DataGenerator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.cheap_model = 'gemini-3.1-flash-lite-preview'
        self.heavy_model = 'gemini-3.1-flash-lite-preview'
        
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7 
        )

    def _call_api_with_retry(self, prompt: str, model_name: str, max_retries: int = 4) -> Optional[list[dict]]:
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self.config
                )
                return json.loads(response.text)
            except Exception as e:
                error_msg = str(e)
                
                # If it's a quota error, wait 65 seconds
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    wait_time = 65
                    print(f"\n  [Rate Limit Hit] API requires a cooldown. Sleeping for {wait_time}s... ({attempt + 1}/{max_retries})")
                else:
                    wait_time = (3 ** attempt) * 3
                    print(f"\n  [Warning] API Error: {e}. Retrying in {wait_time}s... ({attempt + 1}/{max_retries})")
                
                time.sleep(wait_time)
        
        # Crash the function to prevent skipping.
        raise RuntimeError("Max retries exhausted due to API Quota/Errors.")

    def _append_to_jsonl(self, file_path: str, data_pairs: list[dict]):
        if not data_pairs:
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            for pair in data_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # _____________________ Single Chunk Batch Processing ___________________________

    def generate_for_batch(self, chunk_batch: list[Any]) -> list[dict]:
        formatted_chunks = ""
        for chunk in chunk_batch:
            c_id = chunk.metadata.get("chunk_id", "unknown_id")
            formatted_chunks += f'<chunk id="{c_id}">\n{chunk.body}\n</chunk>\n\n'
            
        prompt = textwrap.dedent(f"""
        Bạn là một chuyên gia tạo dữ liệu huấn luyện AI mảng Pháp luật Giao thông Việt Nam.
        Dưới đây là một danh sách các đoạn trích luật, mỗi đoạn được bọc trong thẻ <chunk> 
        và có một ID riêng biệt.

        NHIỆM VỤ:
        Với MỖI đoạn trích, hãy tạo 3 đến 5 câu hỏi thực tế mà người dân sẽ hỏi.
        Câu trả lời cho câu hỏi PHẢI nằm hoàn toàn bên trong đoạn trích đó. Không được 
        trộn lẫn thông tin giữa các chunk với nhau.

        Đa dạng hóa cách hỏi: Tình huống ("Tôi lỡ..."), Định nghĩa, và Từ khóa.
        Trong đó câu hỏi Tình huống ưu tiên nhiều nhất.
                                 
        Trả về DƯỚI DẠNG DANH SÁCH JSON (JSON Array). 
        Bạn BẮT BUỘC phải ghi rõ `chunk_id` tương ứng cho mỗi câu hỏi để tôi biết 
        câu hỏi đó thuộc về đoạn luật nào.
        [
          {{"chunk_id": "khoan_1_part_1", "query": "câu hỏi 1..."}},
          {{"chunk_id": "khoan_1_part_1", "query": "câu hỏi 2..."}},
          {{"chunk_id": "khoan_2", "query": "câu hỏi 1..."}}
        ]

        CÁC ĐOẠN TRÍCH LUẬT:
        {formatted_chunks}
        """)

        generated_queries = self._call_api_with_retry(prompt, model_name=self.cheap_model)
        if not generated_queries:
            return []
            
        training_pairs = []
        body_lookup = {c.metadata.get("chunk_id"): c.body for c in chunk_batch}
        
        for item in generated_queries:
            returned_id = item.get("chunk_id")
            query_text = item.get("query", "")
            
            if returned_id in body_lookup and query_text:
                training_pairs.append({
                    "query": query_text,
                    "context": body_lookup[returned_id]
                })
                
        return training_pairs

    # ________________________ Compound/Crossover Batch Processing __________________

    def generate_compound_batch(self, scenario_batch: list[list[Any]]) -> list[dict]:        
        formatted_scenarios = ""
        scenario_lookup = {}

        # Build the XML tree for multiple scenarios
        for idx, scenario_chunks in enumerate(scenario_batch):
            s_id = f"scenario_{idx}"
            # Save the raw text bodies for mapping later
            scenario_lookup[s_id] = [chunk.body for chunk in scenario_chunks]
            
            formatted_scenarios += f'<scenario id="{s_id}">\n'
            for chunk in scenario_chunks:
                formatted_scenarios += f'  <chunk>\n{chunk.body}\n  </chunk>\n'
            formatted_scenarios += '</scenario>\n\n'
        
        prompt = textwrap.dedent(f"""
        Bạn là chuyên gia luật học. Dưới đây là các nhóm tình huống, mỗi nhóm được bọc trong 
        thẻ <scenario> và có một ID riêng. Bên trong mỗi <scenario> là nhiều đoạn trích luật.

        NHIỆM VỤ:
        Với MỖI <scenario>, hãy tạo ra 2 đến 3 câu hỏi tình huống phức tạp. 
        Trong tình huống này, người vi phạm phải chịu TẤT CẢ các hậu quả pháp lý được nêu 
        trong các <chunk> của scenario đó (ví dụ: vừa phạt tiền, vừa đi tù).
        TUYỆT ĐỐI không trộn lẫn luật của scenario này với scenario khác.
        CÓ THỂ BỎ QUA nếu các scenario KHÔNG THỂ LIÊN QUAN ĐẾN NHAU (hạn chế).

        Trả về DƯỚI DẠNG DANH SÁCH JSON (JSON Array):
        [
          {{"scenario_id": "scenario_0", "query": "câu hỏi phức tạp cho nhóm 0..."}},
          {{"scenario_id": "scenario_1", "query": "câu hỏi phức tạp cho nhóm 1..."}}
        ]

        CÁC NHÓM TÌNH HUỐNG LUẬT:
        {formatted_scenarios}
        """)

        generated_queries = self._call_api_with_retry(prompt, model_name=self.heavy_model)
        if not generated_queries:
            return []
            
        training_pairs = []
        for item in generated_queries:
            s_id = item.get("scenario_id")
            complex_query = item.get("query", "")
            
            if s_id in scenario_lookup and complex_query:
                for chunk_body in scenario_lookup[s_id]:
                    training_pairs.append({
                        "query": complex_query,
                        "context": chunk_body
                    })
                
        return training_pairs

    # __________________ Controller ______________________________

    def start_generate(self, 
                       standard_chunks: list[Any], 
                       crossover_pool: list[Any], 
                       output_file: str, 
                       complex: bool = False,
                       checkpoint_file: str = "checkpoint.json",
                       standard_batch_size: int = 4,
                       compound_batch_size: int = 2,
                       daily_limit: int = 1400,
                       num_crossovers: int = 150,
                       k: int = 4):
        print("\n INITIALIZING DATA GENERATION PIPELINE...\n")
        
        # Ensure output file exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        if not os.path.exists(output_file):
            open(output_file, 'w', encoding='utf-8').close()

        # Load Checkpoint State
        state = {"standard_idx": 0, "crossover_idx": 0}
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    state = json.load(f)
            except json.JSONDecodeError:
                pass

        standard_start = state.get("standard_idx", 0)
        crossover_start = state.get("crossover_idx", 0)

        # --- PHASE 1: STANDARD CHUNKS (Flash Model) ---
        if standard_start < len(standard_chunks) and not complex:
            end_idx = min(standard_start + daily_limit, len(standard_chunks))
            print(f"--- PHASE 1: STANDARD BATCHES ---")
            print(f"Resuming from chunk {standard_start}. Target end: {end_idx}\n")
            
            try:
                for i in range(standard_start, end_idx, standard_batch_size):
                    batch = standard_chunks[i : i + standard_batch_size]
                    print(f"Processing standard batch {i//standard_batch_size + 1} (Chunks {i+1} to {min(i+standard_batch_size, len(standard_chunks))})...", end=" ", flush=True)
                    
                    try:
                        # Attempt to generate
                        pairs = self.generate_for_batch(batch)
                    except RuntimeError as e:
                        # THE EMERGENCY BRAKE
                        print(f"\nFATAL: {e}")
                        print(f"Progress locked safely at chunk {i}. Shutting down to preserve data.")
                        return
                    
                    if pairs:
                        self._append_to_jsonl(output_file, pairs)
                        print(f"[Generated {len(pairs)} pairs]")
                    else:
                        print("[Failed/Skipped]")
                        
                    # Save checkpoint
                    state["standard_idx"] = i + len(batch) + 1
                    with open(checkpoint_file, 'w') as f:
                        json.dump(state, f)
                        
                    time.sleep(3)
            except KeyboardInterrupt:
                print("\n⏸ Script paused manually. Progress saved.")
                return
        else:
            print("--- PHASE 1: STANDARD BATCHES (COMPLETE) ---\n")

        # --- PHASE 2: CROSSOVER CHUNKS (Pro Model) ---
        if not complex:
            if standard_start >= len(standard_chunks) and crossover_start < num_crossovers:
                print(f"--- PHASE 2: CROSSOVER BATCHES ---")
                print(f"Resuming from crossover {crossover_start}. Target end: {num_crossovers}\n")
                
                # Pre-build all random scenarios to keep them consistent
                random.seed(42) # Keep seed fixed so checkpointing logic stays stable
                all_scenarios = []
                sampled_traffic = random.sample(standard_chunks, min(num_crossovers, len(standard_chunks)))
                
                for traffic_chunk in sampled_traffic:
                    cross_chunk = random.choice(crossover_pool)
                    # A scenario is a list of Chunk objects
                    all_scenarios.append([traffic_chunk, cross_chunk])

                try:
                    for i in range(crossover_start, num_crossovers, compound_batch_size):
                        batch = all_scenarios[i : i + compound_batch_size]
                        print(f"Processing crossover batch {i//compound_batch_size + 1} (Scenarios {i+1} to {min(i+compound_batch_size, num_crossovers)})...", end=" ", flush=True)
                        
                        try:
                            pairs = self.generate_compound_batch(batch)
                        except RuntimeError as e:
                            print(f"\n FATAL: {e}")
                            print(f"Progress locked safely at crossover {i}. Shutting down to preserve data.")
                            return
                        
                        if pairs:
                            self._append_to_jsonl(output_file, pairs)
                            print(f"[Generated {len(pairs)} pairs]")
                        else:
                            print("[Failed/Skipped]")
                            
                        # Save checkpoint
                        state["crossover_idx"] = i + len(batch)
                        with open(checkpoint_file, 'w') as f:
                            json.dump(state, f)
                            
                        time.sleep(5) 
                except KeyboardInterrupt:
                    print("\nScript paused manually. Progress saved.")
                    return
            elif crossover_start >= num_crossovers:
                print("--- PHASE 2: CROSSOVER BATCHES (COMPLETE) ---\n")
            else:
                print("--- PHASE 2: CROSSOVER BATCHES (WAITING FOR STANDARD PHASE) ---\n")

        # --- PHASE 3: COMBINING MULTIPLE CHUNKS (Complex Model) ---
        if complex:
            if crossover_start < num_crossovers:
                print(f"--- PHASE 3: COMBINING MULTIPLE CHUNKS ---")
                print(f"Resuming from crossover {crossover_start}. Target end: {num_crossovers}\n")
                
                # Pre-build all random scenarios to keep them consistent
                random.seed(42) # Keep seed fixed so checkpointing logic stays stable
                all_scenarios = []
                for cross_chunk in crossover_pool:
                    sampled_traffics = random.sample(standard_chunks, k)
                    combined_pool = [cross_chunk]
                    for sampled_traffic in sampled_traffics:
                        combined_pool.append(sampled_traffic)
                    all_scenarios.append(combined_pool)

                try:
                    for i in range(crossover_start, num_crossovers, compound_batch_size):
                        batch = all_scenarios[i : i + compound_batch_size]
                        print(f"Processing combining batch {i//compound_batch_size + 1} (Scenarios {i+1} to {min(i+compound_batch_size, num_crossovers)})...", end=" ", flush=True)
                        
                        try:
                            pairs = self.generate_compound_batch(batch)
                        except RuntimeError as e:
                            print(f"\n FATAL: {e}")
                            print(f"Progress locked safely at crossover {i}. Shutting down to preserve data.")
                            return
                        
                        if pairs:
                            self._append_to_jsonl(output_file, pairs)
                            print(f"[Generated {len(pairs)} pairs]")
                        else:
                            print("[Failed/Skipped]")
                            
                        # Save checkpoint
                        state["crossover_idx"] = i + len(batch)
                        with open(checkpoint_file, 'w') as f:
                            json.dump(state, f)
                            
                        time.sleep(5) 
                except KeyboardInterrupt:
                    print("\nScript paused manually. Progress saved.")
                    return
            elif crossover_start >= num_crossovers:
                print("--- PHASE 3: COMBINING MULTIPLE CHUNKS (COMPLETE) ---\n")

        print("\nALL GENERATION PHASES COMPLETE!")