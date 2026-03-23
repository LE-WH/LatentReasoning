Step 1 — Build the dual  model                                                                                                                                                                                            
                                                                                                                                                                                                                        
This is a one-time setup. It builds the expanded tokenizer + model.                                                                                                                                                      
                                                                                                                                                                                                                        
bash scripts/build_dual_model.sh \                                                                                                                                                                                       
    --base_model Qwen/Qwen2.5-3B-Instruct \                                                                                                                                                                              
    --out_dir ./checkpoints/dual_qwen_3b \                                                                                                                                                                               
    --think_missing add                                                                                                                                                                                                  
                                                                                                                                                                                                                        
What it does internally:                                                                                                                                                                                                 
1. scripts/build_dual_vocab.py — creates the dual tokenizer + dual_vocab_meta.json                                                                                                                                       
2. scripts/expand_model_to_dual_vocab.py — doubles the embedding table (latent copy)                                                                                                                                     
3. scripts/verify_dual_model.py — sanity checks the result                                                                                                                                                               
                                                                                                                                                                                                                        
The final model is saved to ./checkpoints/dual_qwen_3b.                                                                                                                                                                  
                                                                                                                                                                                                                        
---                                                                                                                                                                                                                      
Step 2 — Train                                                                                                                                                                                                           
                                                                                                                                                                                                                        
Use any existing task config, just override model_path to point to the dual model.                                                                                                                                       
                                                                                                                                                                                                                        
MetamathQA (math reasoning, recommended for dual model):                                                                                                                                                                 
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _5_metamathqa \                                                                                                                                                     
    model_path=./checkpoints/dual_qwen_3b \                                                                                                                                                                              
    trainer.experiment_name=dual_metamathqa \                                                                                                                                                                            
    trainer.total_training_steps=400                                                                                                                                                                                     
                                                                                                                                                                                                                        
With latent-only training (loss on think-phase tokens only, like DUALCOT_LATENT_ONLY in scalable-latent-reasoning):                                                                                                      
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _5_metamathqa \                                       
    model_path=./checkpoints/dual_qwen_3b \                                                                
    trainer.experiment_name=dual_metamathqa_latentonly \                                                   
    dual_vocab.latent_only=true \                                                                          
    trainer.total_training_steps=400                                                                       
                                                                                                            
Sokoban or other envs work the same way:                                                                   
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _2_sokoban \                                          
    model_path=./checkpoints/dual_qwen_3b \                                                                
    trainer.experiment_name=dual_sokoban                                                                   
                                                                                                            
---                                                                                                        
Step 3 — Eval                                                                                              
                                                                                                                                                                                                                        
The existing eval script auto-detects the dual model and applies the constraint:                                                                                                                                         
                                                                                                                                                                                                                        
CUDA_VISIBLE_DEVICES=0 python -m ragen.eval \                                                                                                                                                                            
    --config-name eval \                                                                                                                                                                                                 
    model_path=./checkpoints/dual_qwen_3b \                                                                                                                                                                              
    system.CUDA_VISIBLE_DEVICES=0                                                                                                                                                                                        
                                                                                                                                                                                                                        
Or evaluate a trained checkpoint:                                                                                                                                                                                        
CUDA_VISIBLE_DEVICES=0 python -m ragen.eval \                                                                                                                                                                            
    --config-name _5_metamathqa \                                                                                                                                                                                        
    model_path=./checkpoints/dual_metamathqa/global_step_400 \                                                                                                                                                           
    system.CUDA_VISIBLE_DEVICES=0                                                                                                                                                                                        
                                                                                                                                                                                                                        
---                                                                           
                                                      
Key config knobs                                    
                                                            
┌───────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
│                   Override                    │                          Effect                          │
├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤                                                                                                             
│ model_path=./checkpoints/dual_qwen_3b         │ Use the dual model                                       │
├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤                                                                                                             
│ dual_vocab.latent_only=true                   │ Train loss only on latent (think-phase) tokens           │
├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤                                                                                                             
│ actor_rollout_ref.rollout.response_length=800 │ Longer responses (useful since latent tokens are hidden) │
├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤                                                                                                             
│ agent_proxy.enable_think=true                 │ Keep <think> tags enabled (default)                      │
└───────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘                                                                                                             
                                                            
---                                                 
What happens automatically                                 

Once model_path points to a dual model directory (one containing dual_vocab_meta.json):
- Eval/rollout (VllmWrapperWg): the vLLM logits processor is injected — during <think>, only latent tokens [V, V+L) are generated; after </think>, only visible tokens [0, V) are generated                              
- Training (ContextManager): loss mask accounts for latent tokens; with latent_only=true, gradients only flow through the latent token positions              