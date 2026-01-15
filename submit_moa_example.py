import requests
import json
import uuid

# Configuration
SERVER_URL = "http://localhost:8000"

def submit_job():
    job_id = f"moa-example-{uuid.uuid4().hex[:8]}"
    
    # 1. Define the Graph Structure
    # We'll create a simple MoA structure:
    # - Two smaller models (0.6B and 4B) generate initial responses
    # - One larger model (8B) aggregates/refines them
    
    payload = {
        "job_id": job_id,
        
        # Adjacency list: node -> [children]
        "graph": {
            "layer1_qwen2.5_fast": ["layer2_aggregator"],
            "layer1_qwen2.5_thoughtful": ["layer2_aggregator"],
            "layer1_llama3.1_thoughtful": ["layer2_aggregator"],
            "layer1_llama3.2_thoughtful": ["layer2_aggregator"],
            "layer1_llama3.2_fast": ["layer2_aggregator"],
            "layer2_aggregator": []
        },
        
        # Map nodes to specific models
        "node_models": {
            "layer1_qwen2.5_fast": "Qwen/Qwen2.5-3B",
            "layer1_qwen2.5_thoughtful": "Qwen/Qwen2.5-7B",
            "layer1_llama3.1_thoughtful": "meta-llama/Llama-3.1-8B-Instruct",
            "layer1_llama3.2_thoughtful": "meta-llama/Llama-3.2-3B-Instruct",
            "layer1_llama3.2_fast": "meta-llama/Llama-3.2-1B-Instruct",
            "layer2_aggregator": "Qwen/Qwen3-4B-Instruct-2507"
        },
        
        # Initial inputs for source nodes
        "inputs": {
            "layer1_qwen2.5_fast": {
                "messages": [{"role": "user", "content": "Explain quantum entanglement briefly."}],
            },
            "layer1_qwen2.5_thoughtful": {
                "messages": [{"role": "user", "content": "Explain quantum entanglement in detail."}],
            },
            "layer1_llama3.1_thoughtful": {
                "messages": [{"role": "user", "content": "Explain quantum entanglement in detail."}]
            },
            "layer1_llama3.2_thoughtful": {
                "messages": [{"role": "user", "content": "Explain quantum entanglement in detail."}]
            },
            "layer1_llama3.2_fast": {
                "messages": [{"role": "user", "content": "Explain quantum entanglement in detail."}]
            },
            # layer2_aggregator receives inputs automatically from parents
        }
    }

    print(f"Submitting Job {job_id}...")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(f"{SERVER_URL}/submit_graph", json=payload)
        response.raise_for_status()
        result = response.json()
        print("\nSubmission Successful!")
        print(f"Job ID: {result['job_id']}")
        print(f"Sources: {result['sources']}")
        print(f"Total Nodes: {result['num_nodes']}")
    except requests.exceptions.RequestException as e:
        print(f"\nError creating job: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Server response: {e.response.text}")

if __name__ == "__main__":
    for _ in range(128):
        submit_job()
