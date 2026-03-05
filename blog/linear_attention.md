# Blog Post 2: Linear Attention, an Overview




I recently read the Mamba2 paper and is facinated by the idea of state-space duality: how linear attentions relates to a state-space model like Mamba. I intend to give an literature reviews here.



I will update the content in March 2026.



## Plan:

1. Papers: 
   - [x] Linear attention (tranformers are RNNs): the first paper on this
   - [ ] (Opt) performer: random features
   - [ ] Ret net
   - [x] Mamba
   - [x] Mamba2 (State-space duality)
   - [ ] (gated) delta net: solve the 'forget' problem
   - [ ] samba / jamba: combining linear & softmax, sliding window atten?
   - [ ] Liger / LoLCATs: distill a quadratic model (llama3) to linear model
  also
  - [ ] FLA library


2. What I find interesting:
    - [ ] state space duality - how state space models relates to attention
    - [ ] hardware (GPU) aware design
    - [ ] linear attention
    - [ ] why they are not as good as softmax attention
    - [ ] *Code demo, e.g. running a Transformers++ with a few linear attention / state-space models with similar budget and ...

3. TO DO List:

    - [x] Figure out which other papers to read - ask gemini
    - [ ] what are the SOTA?
      - Engineering wise
        - [ ] Qwen-3.5: they use GatedDeltaNet
        - [ ] Jamba
      - Efficiency:
        - [ ] Mamba3
        - [ ] GatedDeltaNet
    - [ ] Choose 3-4 interesting sub topics

4. Topics:
    - [ ] Evolution of linear attentions (or maybe the trend: e.g. there are optimizing for better hardware utilization, ...)
    - [ ] Mamba2: state-space duality -> how they are equivalent
    - [ ] what are SOTA
    - [ ] some kind of code demo? not sure what to do yet:
      - [ ] running them with FLA
      - [ ] KV cache growth
      - [ ] let them produce something on the fly (perhaps to an absurb context window)