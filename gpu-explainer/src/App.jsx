import { useState } from "react";

const stages = [
  {
    id: "prompt",
    label: "OUR PROMPT",
    subtitle: "the spark",
    color: "#E8725A",
    icon: "✏️",
    oneliner: "comfyui is the cockpit. clip + t5 are the translators. our words become coordinates.",
    plain: `we type "a worn photograph of a figure dissolving into teal light, analog grain, CRT static" into ComfyUI. these words are about to become numbers. Flux uses two text encoders: CLIP maps the overall vibe into a single summary vector, while T5-XXL processes each word into a 4,096-number coordinate in meaning-space. "teal" lands near "cyan." "dissolving" lands near "fading." swapping a single word can shift the entire output because it moves the coordinate. when we understand this, we stop prompting like we're describing a picture and start prompting like we're placing pins on a map.`,
    detail: `Flux uses two text encoders. CLIP gives a high-level summary (768 numbers). T5-XXL does the heavy lifting, turning each word into a vector of 4,096 numbers representing meaning. "Teal" becomes a coordinate in meaning-space near "cyan" and "blue-green." "Dissolving" lands near "fading" and "disappearing." Our full prompt becomes a matrix of these vectors. A map of what we want, written in math.`,
    hardware: `Two text encoders run on GPU. CLIP produces a pooled 768-dim summary vector (~400MB weights). T5-XXL produces per-token 4,096-dim embeddings, the main conditioning that guides the image (~4.5GB weights). Tensor cores do the matrix multiplications for both. Output: conditioning tensors sitting in VRAM, ready to guide every denoising step that follows.`
  },
  {
    id: "noise",
    label: "PURE NOISE",
    subtitle: "the blank canvas",
    color: "#3DB8A9",
    icon: "📡",
    oneliner: "the seed is the starting point. the gpu rolls the dice. every image begins as static.",
    plain: `the GPU generates a small block of pure random static. think TV snow. this is our starting point ... a 128×128 grid of random numbers (for a 1024px image). our image is hiding inside this noise like a sculpture inside a block of marble. the model's entire job is to chisel it out. and yes, that seed number you keep rerolling? it determines the exact pattern of this static. same seed, same marble block, same sculpture. this is why locking a seed and changing only the prompt lets us iterate on composition without losing the underlying structure we liked.`,
    detail: `PyTorch calls torch.randn() which triggers CUDA's random number generator across thousands of cores simultaneously. each core generates its own random values. the result is a "latent" tensor, much smaller than the final image because it works in compressed space. a 1024×1024 image is only 128×128×16 channels in latent space. the seed we set in KSampler determines the exact pattern of this noise.`,
    hardware: `CUDA cores (not tensor cores) handle random number generation. each of the 21,760 CUDA cores can generate random values independently. the latent tensor allocates ~4MB in VRAM. tiny compared to the model weights (~22GB for Flux Dev). the seed initializes the pseudorandom number generator state, which is why the same seed gives the same image.`
  },
  {
    id: "guidance",
    label: "GUIDANCE",
    subtitle: "how flux steers differently",
    color: "#B080D0",
    icon: "🎯",
    oneliner: "flux bakes the steering into the model. one pass instead of two. half the compute.",
    plain: `in older models like Stable Diffusion, the GPU had to run the model TWICE per step ... once with our prompt, once without, then compare the two to amplify what our prompt adds. Flux skips this entirely. it bakes guidance directly into the model as a number we feed in alongside our prompt. one pass per step instead of two. same control, half the work. this is why guidance in Flux feels different than CFG in SD. we're not amplifying a comparison anymore ... we're dialing how strongly the model listens to us. lower guidance (2-3) gives the model more creative freedom. higher guidance (4-5) makes it follow our prompt more literally. for our workflow: flux guidance 3.0-4.0 is the sweet spot. below 2.5 it starts ignoring our prompt. above 5.0 it overburns and we lose nuance. we can't push it like SD's CFG 7-12 range because distilled guidance behaves differently.`,
    detail: `traditional CFG (Classifier-Free Guidance) computes: output = uncond + scale × (cond - uncond). this requires two full forward passes per step. flux Dev uses "distilled guidance" where the guidance scale is embedded as a timestep-like conditioning signal fed into the transformer blocks. the model learned during training to internalize the effect of different guidance strengths. our guidance value (typically 3.0-4.0 for Flux) becomes a tensor that modulates attention and feedforward layers from the inside.`,
    hardware: `this is a massive efficiency win on the 5090. traditional CFG at 30 steps = 60 forward passes. flux at 30 steps = 30 forward passes. you're cutting total tensor core utilization roughly in half per generation. memory bandwidth demand drops proportionally since you're not shuttling activation tensors back and forth for a second pass. this is a big part of why Flux feels faster than SD 1.5/SDXL at similar quality.`
  },
  {
    id: "attention",
    label: "ATTENTION",
    subtitle: "where the magic actually lives",
    color: "#E8C84A",
    icon: "🧠",
    oneliner: "every part of the image looks at every other part. this is where coherence comes from.",
    plain: `this is the most important part of the whole process and the hardest to explain simply. during each denoising step, every patch of our image "looks at" every other patch and asks "are we related?" the sky patches learn they're all sky. the figure patches learn they form a body. and our prompt words attend to the image patches too ... "teal" pulls the color channels, "dissolving" softens the edges. this looking-at-everything-else process is attention. it's why AI images have coherent composition instead of random pixel soup. when we understand this, we understand why spatial language in our prompts ("figure in the foreground, mountains receding into distance") gives us better compositions ... we're literally telling the attention mechanism how patches should relate to each other.`,
    detail: `the Flux transformer uses two types of attention. self-attention: each image patch (token) computes Query, Key, Value matrices, then calculates attention scores against all other patches. for a 1024px image with 64×64 = 4,096 patches, that's a 4096×4096 attention matrix per head. cross-attention: the same process but between image patches and our prompt tokens. each image patch attends to each word in our prompt, pulling in the meaning it needs. the attention formula is softmax(QK^T / √d) × V ... three massive matrix multiplications per attention layer, repeated across 19+ transformer blocks per step.`,
    hardware: `attention is where the 5090's tensor cores earn their silicon. each attention layer requires: (1) Q×K^T multiplication: [4096 × 128] × [128 × 4096] = huge matrix op, (2) score × V multiplication: [4096 × 4096] × [4096 × 128] = another huge matrix op. with 24 attention heads across 19 double-stream and 38 single-stream blocks, we're looking at hundreds of massive matrix multiplications PER STEP. the 680 tensor cores (5th gen, across 170 SMs) do 4×4 matrix multiply-accumulate operations in a single clock cycle in FP16. in practice, ComfyUI uses flash attention (or xformers) which never builds the full attention matrix in memory ... it computes it in tiles, trading some recomputation for massive VRAM savings. same math, smarter execution. memory bandwidth (1,792 GB/s) still becomes the bottleneck ... the tensor cores are fast enough, but they need data fed to them fast enough to stay busy. this is why VRAM speed matters as much as compute for image generation.`
  },
  {
    id: "denoise",
    label: "DENOISING",
    subtitle: "the transformation",
    color: "#E8925A",
    icon: "🔄",
    oneliner: "pytorch is where the math happens. each step peels a layer of noise off our image.",
    plain: `over 20-30 steps, the model looks at the noisy image and our prompt, then predicts what noise to remove. each step peels away a thin layer of static. step 1: vague shapes emerge from nothing. step 10: composition and color take form. step 20: structure solidifies. step 30: fine details and textures sharpen. all of this runs through PyTorch, which is the software layer between ComfyUI and the GPU. when we click generate, ComfyUI tells PyTorch what to do. PyTorch translates those instructions into math (matrix multiplications, attention calculations). then PyTorch hands that math to CUDA, which is NVIDIA's software that tells the physical GPU cores exactly what to compute. ComfyUI is the cockpit. PyTorch is the translator. CUDA is the foreman on the factory floor. this is why denoise strength matters in img2img. at 0.7, we skip the first 30% of steps ... the large-scale structure is already set by our input image. at 1.0, it starts from pure noise. early steps decide composition, late steps decide detail. if our compositions are off, more steps won't fix it. change the prompt or seed instead. steps beyond 30 on flux are usually wasted compute.`,
    detail: `each step is a full forward pass through the Flux transformer. the model takes the noisy latent + prompt conditioning + guidance value + timestep and outputs a noise prediction. the network doesn't predict the final image directly ... it predicts what noise is present at this step. subtract that predicted noise (scaled by the sampler) and we get a slightly cleaner latent. the timestep tells the model how noisy the current state is, which changes its behavior. early steps (high noise) focus on large-scale structure. late steps (low noise) focus on fine detail.`,
    hardware: `each of our 30 steps pushes the full Flux model through the tensor cores. that's all 19 double-stream blocks and 38 single-stream blocks, each containing multi-head attention + feedforward layers. for a single 1024px generation at 30 steps, the 5090 executes thousands of large matrix multiplications. at the 5090's ~419 TFLOPS dense FP8 (or ~838 with sparsity), this takes seconds. a CPU would take hours.`
  },
  {
    id: "sampler",
    label: "SAMPLER",
    subtitle: "the navigator",
    color: "#C4A882",
    icon: "🧭",
    oneliner: "the sampler decides how to remove noise. the scheduler decides how much at each step. two levers, not one.",
    plain: `two separate things happen between each denoising step, and most artists treat them as one dropdown. the sampler is the algorithm that decides how to remove noise at each step. it looks at the model's prediction and calculates how much noise to subtract and in what direction. the scheduler is the algorithm that decides the noise levels at each step ... how much total noise should be left after step 1, step 2, step 15, step 30. think of it like driving to a destination. the sampler controls steering: Euler drives in a straight line each step. DPM++ 2M checks the rearview mirror and curves its path based on where it's been. the scheduler controls speed: normal spacing is constant. Karras front-loads the big moves early (when the image is mostly noise and decisions matter most) and eases into smaller refinements at the end. these are two independent creative levers. changing the sampler changes the character of the output. changing the scheduler changes where the effort gets concentrated. experiment with both separately to understand what each one does to our images.`,
    detail: `mathematically, denoising solves a differential equation ... the path from noise to image is a continuous curve in latent space. samplers approximate this curve with discrete steps. Euler: first-order, takes the gradient at current point and steps forward. simple but needs more steps. DPM++ 2M: multi-step method that stores the previous noise prediction and uses both current + previous to estimate a curved path. fewer steps needed for same quality. the scheduler (normal, karras, simple) controls sigma values ... the noise levels at each step. Karras front-loads bigger noise reductions early, leaving finer adjustments for later steps.`,
    hardware: `sampler math is lightweight compared to the transformer pass. it's element-wise operations (add, multiply, subtract on tensors of the same shape). these run on CUDA cores, not tensor cores. tensor cores handle the big matrix multiplications inside the model. CUDA cores handle everything between steps: noise scheduling arithmetic, tensor blending, applying the step update. think of it as tensor cores doing the heavy thinking, CUDA cores doing the bookkeeping. dPM++ 2M uses slightly more VRAM than euler because it stores a buffer of previous noise predictions (~4MB extra). negligible on 32GB.`
  },
  {
    id: "vram",
    label: "VRAM MAP",
    subtitle: "how 32gb gets carved up",
    color: "#6AACB8",
    icon: "💾",
    oneliner: "32gb sounds like a lot. flux dev uses 28-30gb of it. every byte has a job.",
    plain: `the 5090 has 32GB of video memory. think of it as a workspace desk. the model weights (the learned knowledge) take up most of the desk ... about 22GB for Flux Dev (transformer + text encoders + VAE). the latent image being worked on is tiny, under 1MB. but during each step, the attention layers create huge temporary scratchpads (activation memory) that spike to 4-6GB. the VAE decoder needs its own space at the end. everything has to fit on the desk at once or generation fails. this is why we can't just "run a bigger model" ... it's not about compute speed, it's about whether the whole pipeline fits in memory at once. and it's why quantization (FP8, NF4) matters ... it shrinks model weights so we can reclaim desk space for higher resolutions or stacking LoRAs.`,
    detail: `VRAM breakdown for Flux Dev at 1024×1024: the permanent residents are the model weights at ~22GB (transformer ~17GB + text encoders ~5GB + VAE ~0.3GB). the latent image is tiny at ~0.5MB. then there's the scratch paper: during each transformer layer, the model produces temporary intermediate results (activation memory) that spike to 4-6GB. these get created, used, and deleted layer by layer. the attention layers also store "key" and "value" results (~1-2GB) so patches can reference each other. flash attention is a trick that computes attention without ever building the full comparison matrix in memory, saving gigabytes of scratch space. total peak: ~28-30GB of 32GB. this is why Flux Dev fits on a 5090 but struggles on 24GB cards without quantization. FP8 quantization shrinks model weights to ~12GB, freeing space for higher resolutions or stacking LoRAs.`,
    hardware: `the 32GB GDDR7 on the 5090 connects via a 512-bit memory bus at 1,792 GB/s bandwidth. during inference, model weights stay resident in VRAM and get read into the streaming multiprocessors as needed. the L2 cache (98MB on the 5090) helps by keeping frequently accessed weights close, but the full ~22GB of model weights still need to flow through the memory bus over the course of each step. at 1,792 GB/s, this is fast but still the tightest bottleneck in the pipeline. the 5090's GDDR7 is a 77% bandwidth upgrade over the 4090's GDDR6X (1,008 GB/s), which directly translates to faster generation times.`
  },
  {
    id: "resolution",
    label: "RESOLUTION",
    subtitle: "why 2x pixels ≠ 2x time",
    color: "#D86A8A",
    icon: "📐",
    oneliner: "double the resolution. 16x the attention compute. quadratic scaling is brutal.",
    plain: `here's something that surprises everyone. doubling our image from 1024 to 2048 pixels doesn't make it twice as slow. it makes it roughly 16 times slower on the attention layers. why? because every patch has to look at every other patch. double the width and height means 4x the patches. but attention compares every patch to every other patch, so it's 4×4 = 16x the comparisons. the GPU just audibly sighed. this is why high-res generation is so much more expensive than it seems like it should be, and why the 5090's fans spin up like it's preparing for takeoff. this is why the smartest workflow isn't generating at max resolution ... it's generating at 1024, nailing the composition and prompt, then upscaling. we get 90% of the quality at a fraction of the compute. for our workflow: generate at 1024, upscale with a dedicated model. almost nobody generates natively at 2048 because the compute cost is brutal and the quality gain doesn't justify it when good upscalers exist.`,
    detail: `self-attention scales quadratically with token count: o(n²). at 1024×1024, the latent is 64×64 = 4,096 tokens. the attention matrix is 4096×4096 = 16.7 million entries. at 2048×2048, the latent is 128×128 = 16,384 tokens. attention matrix: 16384×16384 = 268 million entries. that's 16x more compute for a 4x increase in pixels. the feedforward layers scale linearly (4x), so total scaling is somewhere between 4x and 16x depending on the model architecture. Flux Dev's deep attention stack means it trends closer to the quadratic end.`,
    hardware: `at 2048×2048, VRAM becomes critical. activation memory for attention scales quadratically too ... we need to store those massive attention matrices. a single attention layer's score matrix goes from ~64MB at 1024px to ~1GB at 2048px. across all heads and layers, peak activation memory can exceed available VRAM. solutions: attention slicing (compute attention in chunks, trading speed for memory), xformers/flash attention (mathematically equivalent but never materializes the full attention matrix), or tiled VAE decoding. the 5090's 32GB gives us more headroom here than the 4090's 24GB, but 2048px on Flux Dev still requires memory optimization.`
  },
  {
    id: "vae",
    label: "VAE DECODE",
    subtitle: "the reveal",
    color: "#3DB8A9",
    icon: "🖼️",
    oneliner: "128×128 compressed → 1024×1024 pixels. the final unzip. our image exists.",
    plain: `the denoised latent is still in compressed space (128×128). the VAE decoder is the final translator ... it unpacks our tiny grid into a full 1024×1024 pixel image with real colors. like decompressing a zip file, but for images. the math is done. the noise is gone. colors, details, textures all expand to their final resolution. our image exists. saved to disk. the 5090 exhales.`,
    detail: `the VAE (Variational Autoencoder) decoder is a convolutional neural network that upscales the latent back to pixel space. it learned during training how to reconstruct full-resolution images from compressed representations. the 16-channel latent at 128×128 becomes a 3-channel RGB image at 1024×1024 (8× spatial upscale). convolutions progressively expand the spatial dimensions while reducing channels. each upsampling block doubles the resolution.`,
    hardware: `one more GPU forward pass, but through a much smaller model than the main transformer (~160MB weights vs ~17GB). convolution operations run on tensor cores. the output image (~12MB for 1024×1024 RGB float32) writes to VRAM, then transfers over PCIe 5.0 to system RAM where ComfyUI saves it as PNG. this final step takes ~50-100ms. total generation time for 30 steps at 1024×1024 on the 5090: roughly 8-15 seconds depending on sampler and precision settings.`
  }
];

const SamplerVisual = () => {
  // Euler: straight line steps. DPM++: curved path
  const steps = 8;
  const w = 380;
  const h = 140;
  const pad = 40;
  const graphW = w - pad * 2;
  const graphH = h - pad - 16;

  // Euler path: linear steps down
  const eulerPts = Array.from({ length: steps + 1 }, (_, i) => {
    const x = pad + (i / steps) * graphW;
    const y = 16 + (1 - (1 - i / steps)) * graphH;
    return `${x},${y}`;
  }).join(" ");

  // DPM++ path: curved (gets closer faster)
  const dpmPts = Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    const curve = 1 - Math.pow(1 - t, 1.8);
    const x = pad + t * graphW;
    const y = 16 + (1 - (1 - curve)) * graphH;
    return `${x},${y}`;
  }).join(" ");

  // Karras schedule: big drops early, gentle late
  const karrasNoise = Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    const noise = Math.pow(1 - t, 2.2);
    const x = pad + t * graphW;
    const y = 16 + (1 - noise) * graphH;
    return `${x},${y}`;
  }).join(" ");

  // Normal schedule: constant drops
  const normalNoise = Array.from({ length: steps + 1 }, (_, i) => {
    const x = pad + (i / steps) * graphW;
    const y = 16 + (i / steps) * graphH;
    return `${x},${y}`;
  }).join(" ");

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {/* Sampler comparison */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          borderRadius: 10,
          padding: "12px 8px 8px",
          border: "1px solid rgba(196,168,130,0.15)"
        }}>
          <div style={{
            fontSize: 10,
            color: "#C4A882",
            fontFamily: "'JetBrains Mono', monospace",
            textAlign: "center",
            marginBottom: 4,
            letterSpacing: 1
          }}>
            sampler: how we walk
          </div>
          <svg width="100%" viewBox={`0 0 ${w} ${h}`}>
            {/* Axis labels */}
            <text x={pad - 4} y={14} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">noise</text>
            <text x={pad - 4} y={h - pad + 14} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">clean</text>
            <text x={pad} y={h - pad + 28} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace">step 1</text>
            <text x={w - pad} y={h - pad + 28} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">step {steps}</text>
            {/* Axis lines */}
            <line x1={pad} y1={16} x2={pad} y2={h - pad} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
            <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
            {/* Euler: straight */}
            <polyline points={eulerPts} fill="none" stroke="#E8725A" strokeWidth={2} opacity={0.8} />
            {/* DPM++: curved */}
            <polyline points={dpmPts} fill="none" stroke="#3DB8A9" strokeWidth={2} opacity={0.8} strokeDasharray="6 3" />
            {/* Legend */}
            <line x1={pad + 10} y1={h - 8} x2={pad + 28} y2={h - 8} stroke="#E8725A" strokeWidth={2} />
            <text x={pad + 32} y={h - 5} fill="#bbb" fontSize="8" fontFamily="'JetBrains Mono', monospace">euler (straight)</text>
            <line x1={pad + 150} y1={h - 8} x2={pad + 168} y2={h - 8} stroke="#3DB8A9" strokeWidth={2} strokeDasharray="6 3" />
            <text x={pad + 172} y={h - 5} fill="#bbb" fontSize="8" fontFamily="'JetBrains Mono', monospace">dpm++ (curved)</text>
          </svg>
        </div>

        {/* Scheduler comparison */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          borderRadius: 10,
          padding: "12px 8px 8px",
          border: "1px solid rgba(196,168,130,0.15)"
        }}>
          <div style={{
            fontSize: 10,
            color: "#C4A882",
            fontFamily: "'JetBrains Mono', monospace",
            textAlign: "center",
            marginBottom: 4,
            letterSpacing: 1
          }}>
            scheduler: where effort lands
          </div>
          <svg width="100%" viewBox={`0 0 ${w} ${h}`}>
            {/* Axis labels */}
            <text x={pad - 4} y={14} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">high</text>
            <text x={pad - 4} y={h - pad + 14} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">low</text>
            <text x={pad} y={h - pad + 28} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace">step 1</text>
            <text x={w - pad} y={h - pad + 28} fill="#888" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">step {steps}</text>
            {/* Axis lines */}
            <line x1={pad} y1={16} x2={pad} y2={h - pad} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
            <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
            {/* Normal: linear */}
            <polyline points={normalNoise} fill="none" stroke="#E8725A" strokeWidth={2} opacity={0.8} />
            {/* Karras: steep early, gentle late */}
            <polyline points={karrasNoise} fill="none" stroke="#B080D0" strokeWidth={2} opacity={0.8} strokeDasharray="6 3" />
            {/* Legend */}
            <line x1={pad + 10} y1={h - 8} x2={pad + 28} y2={h - 8} stroke="#E8725A" strokeWidth={2} />
            <text x={pad + 32} y={h - 5} fill="#bbb" fontSize="8" fontFamily="'JetBrains Mono', monospace">normal (even)</text>
            <line x1={pad + 150} y1={h - 8} x2={pad + 168} y2={h - 8} stroke="#B080D0" strokeWidth={2} strokeDasharray="6 3" />
            <text x={pad + 172} y={h - 5} fill="#bbb" fontSize="8" fontFamily="'JetBrains Mono', monospace">karras (front-loaded)</text>
          </svg>
        </div>
      </div>
      <div style={{
        fontSize: 10,
        color: "#999",
        fontFamily: "'JetBrains Mono', monospace",
        textAlign: "center",
        marginTop: 8,
        lineHeight: 1.6
      }}>
        same number of steps, same compute cost. the difference is where the work lands.
      </div>
    </div>
  );
};

const VRAMBar = () => {
  const segments = [
    { label: "flux transformer", gb: 17, color: "#E8725A" },
    { label: "text encoders (clip + t5)", gb: 5, color: "#B080D0" },
    { label: "VAE", gb: 0.3, color: "#3DB8A9" },
    { label: "activations (peak)", gb: 5, color: "#E8C84A" },
    { label: "kv cache", gb: 1.5, color: "#6AACB8" },
    { label: "latent + misc", gb: 0.5, color: "#C4A882" },
  ];
  const total = 32;
  const used = segments.reduce((s, x) => s + x.gb, 0);

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{
        display: "flex",
        height: 28,
        borderRadius: 6,
        overflow: "hidden",
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.1)"
      }}>
        {segments.map((seg, i) => (
          <div
            key={i}
            style={{
              width: `${(seg.gb / total) * 100}%`,
              background: seg.color + "55",
              borderRight: i < segments.length - 1 ? "1px solid rgba(0,0,0,0.3)" : "none",
              position: "relative",
              transition: "all 0.3s ease"
            }}
            title={`${seg.label}: ${seg.gb}GB`}
          />
        ))}
        <div style={{
          width: `${((total - used) / total) * 100}%`,
          background: "rgba(255,255,255,0.02)"
        }} title={`Free: ${(total - used).toFixed(1)}GB`} />
      </div>
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "8px 16px",
        marginTop: 10
      }}>
        {segments.map((seg, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{
              width: 8,
              height: 8,
              borderRadius: 2,
              background: seg.color + "88"
            }} />
            <span style={{
              fontSize: 10,
              color: "#bbb",
              fontFamily: "'JetBrains Mono', monospace"
            }}>
              {seg.label} ({seg.gb}GB)
            </span>
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(255,255,255,0.1)" }} />
          <span style={{ fontSize: 10, color: "#d5d5d5", fontFamily: "'JetBrains Mono', monospace" }}>
            Free ({(total - used).toFixed(1)}GB)
          </span>
        </div>
      </div>
    </div>
  );
};

const ResolutionScale = () => {
  const data = [
    { res: "512", patches: 1024, attn: "1M", relative: 1, bar: 2 },
    { res: "1024", patches: 4096, attn: "16.7M", relative: 16, bar: 16 },
    { res: "1536", patches: 9216, attn: "84.9M", relative: 83, bar: 50 },
    { res: "2048", patches: 16384, attn: "268M", relative: 262, bar: 100 },
  ];

  return (
    <div style={{ marginTop: 4 }}>
      {data.map((d, i) => (
        <div key={i} style={{
          display: "grid",
          gridTemplateColumns: "60px 1fr 70px",
          alignItems: "center",
          gap: 12,
          marginBottom: 8
        }}>
          <span style={{
            fontSize: 11,
            color: "#e0e0e0",
            fontFamily: "'JetBrains Mono', monospace",
            textAlign: "right"
          }}>
            {d.res}px
          </span>
          <div style={{
            height: 16,
            borderRadius: 4,
            background: `linear-gradient(90deg, #D86A8A55, #D86A8A22)`,
            width: `${d.bar}%`,
            transition: "width 0.5s ease",
            minWidth: 4
          }} />
          <span style={{
            fontSize: 10,
            color: "#bbb",
            fontFamily: "'JetBrains Mono', monospace"
          }}>
            {d.relative}×
          </span>
        </div>
      ))}
      <div style={{
        fontSize: 10,
        color: "#aaa",
        fontFamily: "'JetBrains Mono', monospace",
        marginTop: 8,
        textAlign: "center"
      }}>
        attention compute cost (relative to 512px)
      </div>
    </div>
  );
};

const FlowDiagram = () => {
  const nodes = [
    { label: "our prompt", sub: "text string", color: "#E8725A", hw: "cpu → gpu" },
    { label: "clip + t5 encoders", sub: "words → vectors", color: "#E8725A", hw: "tensor cores" },
    { label: "conditioning tensors", sub: "in vram", color: "#B080D0", hw: "~5gb" },
    { label: "torch.randn()", sub: "noise from seed", color: "#3DB8A9", hw: "cuda cores" },
    { label: "noisy latent", sub: "128×128×16", color: "#3DB8A9", hw: "~0.5mb" },
    { label: "flux transformer", sub: "57 blocks", color: "#E8925A", hw: "tensor cores" },
    { label: "attention", sub: "self + cross (flash)", color: "#E8C84A", hw: "tensor cores" },
    { label: "feedforward", sub: "transform", color: "#E8925A", hw: "tensor cores" },
    { label: "noise prediction", sub: "what to remove", color: "#C4A882", hw: "vram" },
    { label: "sampler", sub: "euler / dpm++", color: "#C4A882", hw: "cuda cores" },
    { label: "denoised latent", sub: "128×128 clean", color: "#6AACB8", hw: "vram" },
    { label: "vae decoder", sub: "decompress", color: "#3DB8A9", hw: "tensor cores" },
    { label: "1024×1024 image", sub: "png saved", color: "#3DB8A9", hw: "disk" },
  ];

  const loopStart = 5;
  const loopEnd = 9;

  const nodeH = 44;
  const arrowH = 24;
  const totalH = nodes.length * nodeH + (nodes.length - 1) * arrowH;
  const loopX = 420;

  const getY = (i) => i * (nodeH + arrowH);

  return (
    <div style={{ marginTop: 4, position: "relative", overflowX: "auto" }}>
      <svg width="100%" viewBox={`0 0 480 ${totalH + 20}`} style={{ minWidth: 400 }}>
        {/* Loop bracket on the right side */}
        <line
          x1={loopX} y1={getY(loopStart) + nodeH / 2}
          x2={loopX} y2={getY(loopEnd) + nodeH / 2}
          stroke="#E8925A" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.5"
        />
        <line
          x1={loopX - 8} y1={getY(loopStart) + nodeH / 2}
          x2={loopX} y2={getY(loopStart) + nodeH / 2}
          stroke="#E8925A" strokeWidth="1.5" opacity="0.5"
        />
        <line
          x1={loopX - 8} y1={getY(loopEnd) + nodeH / 2}
          x2={loopX} y2={getY(loopEnd) + nodeH / 2}
          stroke="#E8925A" strokeWidth="1.5" opacity="0.5"
        />
        {/* Loop arrow pointing back up */}
        <polygon
          points={`${loopX - 4},${getY(loopStart) + nodeH / 2 + 6} ${loopX},${getY(loopStart) + nodeH / 2 - 2} ${loopX + 4},${getY(loopStart) + nodeH / 2 + 6}`}
          fill="#E8925A" opacity="0.6"
        />
        <text
          x={loopX + 8} y={getY(loopStart) + (getY(loopEnd) - getY(loopStart)) / 2 + nodeH / 2 + 4}
          fill="#E8925A" fontSize="9" fontFamily="'JetBrains Mono', monospace" opacity="0.7"
          transform={`rotate(90, ${loopX + 8}, ${getY(loopStart) + (getY(loopEnd) - getY(loopStart)) / 2 + nodeH / 2 + 4})`}
        >
          × 30 steps
        </text>

        {nodes.map((node, i) => {
          const y = getY(i);
          const inLoop = i >= loopStart && i <= loopEnd;

          return (
            <g key={i}>
              {/* Node background */}
              <rect
                x={8} y={y}
                width={400} height={nodeH}
                rx={8} ry={8}
                fill={inLoop ? "rgba(232, 146, 90, 0.06)" : "rgba(255,255,255,0.02)"}
                stroke={node.color} strokeWidth={0.5} strokeOpacity={0.3}
              />

              {/* Colored dot */}
              <circle
                cx={24} cy={y + nodeH / 2}
                r={4} fill={node.color}
              />

              {/* Label */}
              <text
                x={38} y={y + 18}
                fill="#e0e0e0" fontSize="11.5"
                fontFamily="'JetBrains Mono', monospace" fontWeight="600"
              >
                {node.label}
              </text>

              {/* Subtitle */}
              <text
                x={38} y={y + 33}
                fill="#999" fontSize="9.5"
                fontFamily="'JetBrains Mono', monospace"
              >
                {node.sub}
              </text>

              {/* Hardware label right-aligned */}
              <text
                x={396} y={y + nodeH / 2 + 4}
                fill="#bbb" fontSize="9"
                fontFamily="'JetBrains Mono', monospace"
                textAnchor="end"
              >
                {node.hw}
              </text>

              {/* Arrow to next node */}
              {i < nodes.length - 1 && (
                <g>
                  <line
                    x1={24} y1={y + nodeH}
                    x2={24} y2={y + nodeH + arrowH}
                    stroke={inLoop || (i + 1 >= loopStart && i + 1 <= loopEnd) ? "#E8925A" : "rgba(255,255,255,0.15)"}
                    strokeWidth={inLoop ? 1.5 : 1}
                  />
                  <polygon
                    points={`${24 - 3},${y + nodeH + arrowH - 5} ${24},${y + nodeH + arrowH} ${24 + 3},${y + nodeH + arrowH - 5}`}
                    fill={inLoop || (i + 1 >= loopStart && i + 1 <= loopEnd) ? "#E8925A" : "rgba(255,255,255,0.2)"}
                  />
                </g>
              )}
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{
        marginTop: 12,
        padding: "10px 14px",
        background: "rgba(255,255,255,0.02)",
        borderRadius: 8,
        display: "flex",
        justifyContent: "center",
        gap: 20,
        flexWrap: "wrap"
      }}>
        {[
          { color: "#E8C84A", label: "attention (heaviest)" },
          { color: "#E8925A", label: "denoising loop" },
          { color: "#C4A882", label: "sampler (lightweight)" },
          { color: "#3DB8A9", label: "encode / decode" },
        ].map((leg, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: leg.color }} />
            <span style={{ fontSize: 9, color: "#aaa", fontFamily: "'JetBrains Mono', monospace" }}>
              {leg.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const Takeaways = () => {
  const items = [
    {
      num: "01",
      title: "prompts are coordinates, not descriptions",
      text: "each word in our prompt gets converted to a point in a 4,096-dimensional meaning-space (via T5-XXL). words with similar meanings land near each other. 'dissolving' and 'fading' are neighbors. 'dissolving' and 'brick' are far apart. the model uses those distances to decide how concepts blend visually. we can't see this space directly, but we learn it through iteration: swap one word, see what shifts, build intuition about which words pull similar results and which ones diverge.",
      color: "#E8725A"
    },
    {
      num: "02",
      title: "early steps = structure, late steps = detail",
      text: "the denoising process front-loads composition decisions. by step 10, the layout is largely set. this is why denoise strength in img2img is so powerful, and why adding steps past 25-30 has diminishing returns. we can use fewer steps for drafts and save compute for finals.",
      color: "#3DB8A9"
    },
    {
      num: "03",
      title: "resolution scaling is quadratic, not linear",
      text: "going from 1024 to 2048 isn't 2x or even 4x the compute on attention layers. it's 16x. the smartest artists generate at 1024, lock their composition, then upscale. understanding this one fact saves hours of gpu time every week.",
      color: "#D86A8A"
    },
    {
      num: "04",
      title: "vram bandwidth is the real bottleneck, not compute",
      text: "the 5090's tensor cores can crunch numbers faster than memory can feed them data. generation speed is limited by how fast ~22gb of model weights can be read from vram each step (1,792 gb/s). both compute power and memory speed matter, but for image generation, faster vram (gddr7) often has a more noticeable impact on generation time than adding more cores.",
      color: "#6AACB8"
    },
    {
      num: "05",
      title: "the sampler and scheduler are separate creative controls",
      text: "the sampler decides the path through latent space (euler = straight, dpm++ = curved). the scheduler decides the pace (karras = aggressive early, gentle late). most artists treat these as one dropdown. they're two independent levers that compound. understanding what each one does lets you make intentional choices instead of guessing.",
      color: "#C4A882"
    }
  ];

  return (
    <div style={{ marginTop: 4 }}>
      {items.map((item, i) => (
        <div key={i} style={{
          display: "flex",
          gap: 14,
          padding: "14px 0",
          borderBottom: i < items.length - 1 ? "1px solid rgba(255,255,255,0.05)" : "none"
        }}>
          <div style={{
            fontSize: 20,
            fontWeight: 200,
            color: item.color,
            fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 1,
            flexShrink: 0,
            width: 28
          }}>
            {item.num}
          </div>
          <div>
            <div style={{
              fontSize: 13,
              fontWeight: 600,
              color: "#e0e0e0",
              marginBottom: 6
            }}>
              {item.title}
            </div>
            <div style={{
              fontSize: 12,
              color: "#bbb",
              lineHeight: 1.7
            }}>
              {item.text}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const CoreComparison = () => {
  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div style={{
          background: "rgba(61, 184, 169, 0.06)",
          border: "1px solid rgba(61, 184, 169, 0.15)",
          borderRadius: 10,
          padding: 18
        }}>
          <div style={{ fontSize: 24, marginBottom: 6 }}>⚡</div>
          <div style={{
            fontSize: 13,
            fontWeight: 700,
            color: "#3DB8A9",
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 10
          }}>
            21,760 CUDA CORES
          </div>
          <div style={{ fontSize: 12, color: "#e0e0e0", lineHeight: 1.7 }}>
            The <span style={{ color: "#bbb" }}>general workers</span>. Each one does simple math: add, multiply, compare. One operation at a time, but thousands running simultaneously.
          </div>
          <div style={{
            marginTop: 12,
            padding: "8px 12px",
            background: "rgba(61, 184, 169, 0.05)",
            borderRadius: 6,
            fontSize: 11,
            color: "#8eecd5",
            lineHeight: 1.6
          }}>
            noise generation, sampler math, activation functions, everything that isn't matrix multiply
          </div>
          <div style={{
            marginTop: 8,
            fontSize: 11,
            color: "#aaa",
            fontStyle: "italic"
          }}>
            21,760 workers each carrying one brick, incredibly fast
          </div>
        </div>

        <div style={{
          background: "rgba(232, 114, 90, 0.06)",
          border: "1px solid rgba(232, 114, 90, 0.15)",
          borderRadius: 10,
          padding: 18
        }}>
          <div style={{ fontSize: 24, marginBottom: 6 }}>🔥</div>
          <div style={{
            fontSize: 13,
            fontWeight: 700,
            color: "#E8725A",
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 10
          }}>
            680 TENSOR CORES
          </div>
          <div style={{ fontSize: 12, color: "#e0e0e0", lineHeight: 1.7 }}>
            The <span style={{ color: "#bbb" }}>heavy lifters</span>. Each multiplies entire 4×4 matrices in a single clock cycle. Specialized for exactly the math neural networks need most.
          </div>
          <div style={{
            marginTop: 12,
            padding: "8px 12px",
            background: "rgba(232, 114, 90, 0.05)",
            borderRadius: 6,
            fontSize: 11,
            color: "#f0a898",
            lineHeight: 1.6
          }}>
            attention layers, linear projections, convolutions ... 90%+ of model compute
          </div>
          <div style={{
            marginTop: 8,
            fontSize: 11,
            color: "#aaa",
            fontStyle: "italic"
          }}>
            680 cranes each moving an entire pallet of bricks at once
          </div>
        </div>
      </div>

      <div style={{
        marginTop: 16,
        padding: "12px 16px",
        background: "rgba(196, 168, 130, 0.06)",
        border: "1px solid rgba(196, 168, 130, 0.1)",
        borderRadius: 8,
        fontSize: 11,
        color: "#C4A882",
        lineHeight: 1.7,
        textAlign: "center"
      }}>
        each denoising step: tensor cores do ~90% of compute (matrix math)
        <br />
        cuda cores handle ~10% (sampler logic, noise scheduling, activations)
        <br />
        both share the same 32gb vram pool and 1,792 gb/s memory bus
      </div>
    </div>
  );
};

export default function GPUExplainer() {
  const [activeStage, setActiveStage] = useState(0);
  const [depth, setDepth] = useState("plain");
  const [expandedPanel, setExpandedPanel] = useState(null);

  const stage = stages[activeStage];

  const panels = [
    { id: "takeaways", label: "💡 5 things most artists don't know", component: <Takeaways /> },
    { id: "flow", label: "🔀 full hardware flow diagram", component: <FlowDiagram /> },
    { id: "sampler", label: "🧭 sampler vs scheduler (visual)", component: <SamplerVisual /> },
    { id: "cores", label: "⚡ cuda cores vs 🔥 tensor cores", component: <CoreComparison /> },
    { id: "vram", label: "💾 vram allocation map (32gb)", component: <VRAMBar /> },
    { id: "resolution", label: "📐 resolution scaling (quadratic pain)", component: <ResolutionScale /> },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#09090b",
      color: "#e5e5e5",
      fontFamily: "'Inter', -apple-system, sans-serif",
      padding: "40px 20px",
      display: "flex",
      justifyContent: "center"
    }}>
      <div style={{ maxWidth: 720, width: "100%" }}>
        {/* Header */}
        <div style={{ marginBottom: 40 }}>
          <div style={{
            fontSize: 15,
            color: "#e8e8e8",
            marginBottom: 20,
            lineHeight: 1.6,
            fontStyle: "italic"
          }}>
            what the f*ck is really happening when we type a prompt in and change the settings
          </div>
          <div style={{
            fontSize: 11,
            color: "#aaa",
            marginBottom: 16,
            lineHeight: 1.6
          }}>
            after 1000+ generations on a local flux dev pipeline, i mapped every stage from prompt to pixel ... what the software does, what the hardware does, and what it means for our creative control
          </div>
          <div style={{
            fontSize: 9,
            letterSpacing: 5,
            color: "#aaa",
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 10,
            textTransform: "uppercase"
          }}>
            from prompt to pixel
          </div>
          <h1 style={{
            fontSize: 30,
            fontWeight: 200,
            color: "#f5f5f5",
            margin: 0,
            lineHeight: 1.3,
            letterSpacing: -0.5
          }}>
            how our words become images
          </h1>
          <div style={{
            fontSize: 12,
            color: "#bbb",
            marginTop: 10,
            fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: 0.5
          }}>
            flux dev · pytorch · rtx 5090
          </div>
          <div style={{
            fontSize: 12,
            color: "#999",
            marginTop: 16,
            lineHeight: 1.7,
            borderTop: "1px solid rgba(255,255,255,0.06)",
            paddingTop: 16
          }}>
            i run flux dev on a 5090 with comfyui every day. i wanted to understand what my machine is actually doing when i change the settings. not the surface level stuff. the real path from prompt to pixel, mapped to the physical hardware. this is what i found.
          </div>
        </div>

        {/* Stage Navigation */}
        <div style={{
          display: "flex",
          gap: 1,
          marginBottom: 28,
          overflowX: "auto",
          paddingBottom: 4
        }}>
          {stages.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setActiveStage(i)}
              style={{
                flex: "1 0 auto",
                minWidth: 64,
                padding: "12px 6px 8px",
                background: i === activeStage ? "rgba(255,255,255,0.08)" : "transparent",
                border: "none",
                borderBottom: `2px solid ${i === activeStage ? s.color : "rgba(255,255,255,0.08)"}`,
                cursor: "pointer",
                transition: "all 0.25s ease"
              }}
            >
              <div style={{ fontSize: 16, marginBottom: 3 }}>{s.icon}</div>
              <div style={{
                fontSize: 8,
                letterSpacing: 1.2,
                color: i === activeStage ? s.color : "#444",
                fontFamily: "'JetBrains Mono', monospace",
                transition: "color 0.25s ease",
                whiteSpace: "nowrap"
              }}>
                {s.label}
              </div>
            </button>
          ))}
        </div>

        {/* Depth Toggle */}
        <div style={{
          display: "flex",
          gap: 3,
          marginBottom: 20,
          background: "rgba(255,255,255,0.02)",
          borderRadius: 8,
          padding: 3,
          width: "fit-content"
        }}>
          {[
            { key: "plain", label: "plain english" },
            { key: "detail", label: "technical" },
            { key: "hardware", label: "on the 5090" }
          ].map(d => (
            <button
              key={d.key}
              onClick={() => setDepth(d.key)}
              style={{
                padding: "7px 14px",
                fontSize: 10,
                letterSpacing: 1.2,
                fontFamily: "'JetBrains Mono', monospace",
                background: depth === d.key ? "rgba(255,255,255,0.07)" : "transparent",
                color: depth === d.key ? "#f0f0f0" : "#444",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                transition: "all 0.2s ease",
                textTransform: "lowercase"
              }}
            >
              {d.label}
            </button>
          ))}
        </div>

        {/* Content Card */}
        <div style={{
          background: "rgba(255,255,255,0.015)",
          border: `1px solid rgba(${parseInt(stage.color.slice(1,3),16)},${parseInt(stage.color.slice(3,5),16)},${parseInt(stage.color.slice(5,7),16)},0.1)`,
          borderRadius: 14,
          padding: 28,
          marginBottom: 20,
          minHeight: 180,
          transition: "border-color 0.3s ease"
        }}>
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginBottom: 18
          }}>
            <div style={{
              width: 34,
              height: 34,
              borderRadius: 8,
              background: `rgba(${parseInt(stage.color.slice(1,3),16)},${parseInt(stage.color.slice(3,5),16)},${parseInt(stage.color.slice(5,7),16)},0.07)`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 17
            }}>
              {stage.icon}
            </div>
            <div>
              <div style={{
                fontSize: 15,
                fontWeight: 600,
                color: stage.color,
                textTransform: "lowercase"
              }}>
                {stage.label}
              </div>
              <div style={{
                fontSize: 10,
                color: "#d5d5d5",
                fontFamily: "'JetBrains Mono', monospace"
              }}>
                {stage.subtitle}
              </div>
            </div>
            <div style={{
              marginLeft: "auto",
              fontSize: 10,
              color: "#e0e0e0",
              fontFamily: "'JetBrains Mono', monospace"
            }}>
              {activeStage + 1}/{stages.length}
            </div>
          </div>

          {/* One-liner */}
          <div style={{
            fontSize: 12,
            color: stage.color,
            fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 1.6,
            marginBottom: 16,
            padding: "10px 14px",
            background: `rgba(${parseInt(stage.color.slice(1,3),16)},${parseInt(stage.color.slice(3,5),16)},${parseInt(stage.color.slice(5,7),16)},0.04)`,
            borderLeft: `2px solid ${stage.color}`,
            borderRadius: "0 6px 6px 0",
            opacity: 0.9
          }}>
            {stage.oneliner}
          </div>

          <div style={{
            fontSize: 13.5,
            lineHeight: 1.85,
            color: "#d5d5d5"
          }}>
            {stage[depth]}
          </div>
        </div>

        {/* Navigation */}
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 36 }}>
          <button
            onClick={() => setActiveStage(Math.max(0, activeStage - 1))}
            disabled={activeStage === 0}
            style={{
              padding: "9px 18px",
              fontSize: 10,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: 1,
              background: activeStage === 0 ? "transparent" : "rgba(255,255,255,0.03)",
              color: activeStage === 0 ? "#222" : "#666",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 7,
              cursor: activeStage === 0 ? "default" : "pointer"
            }}
          >
            ← prev
          </button>
          <button
            onClick={() => setActiveStage(Math.min(stages.length - 1, activeStage + 1))}
            disabled={activeStage === stages.length - 1}
            style={{
              padding: "9px 18px",
              fontSize: 10,
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: 1,
              background: activeStage === stages.length - 1 ? "transparent" : "rgba(255,255,255,0.03)",
              color: activeStage === stages.length - 1 ? "#222" : "#666",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 7,
              cursor: activeStage === stages.length - 1 ? "default" : "pointer"
            }}
          >
            next →
          </button>
        </div>

        {/* Expandable Panels */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 32 }}>
          {panels.map(panel => (
            <div key={panel.id}>
              <button
                onClick={() => setExpandedPanel(expandedPanel === panel.id ? null : panel.id)}
                style={{
                  width: "100%",
                  padding: "14px 18px",
                  background: expandedPanel === panel.id ? "rgba(255,255,255,0.03)" : "rgba(255,255,255,0.015)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: expandedPanel === panel.id ? "10px 10px 0 0" : 10,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  transition: "all 0.2s ease"
                }}
              >
                <span style={{
                  fontSize: 11,
                  letterSpacing: 1.5,
                  color: "#e0e0e0",
                  fontFamily: "'JetBrains Mono', monospace",
                  textTransform: "lowercase"
                }}>
                  {panel.label}
                </span>
                <span style={{ color: "#d5d5d5", fontSize: 16, fontWeight: 300 }}>
                  {expandedPanel === panel.id ? "−" : "+"}
                </span>
              </button>
              {expandedPanel === panel.id && (
                <div style={{
                  borderLeft: "1px solid rgba(255,255,255,0.08)",
                  borderRight: "1px solid rgba(255,255,255,0.08)",
                  borderBottom: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "0 0 10px 10px",
                  padding: "16px 18px 20px"
                }}>
                  {panel.component}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Flow Summary */}
        <div style={{
          padding: "18px 22px",
          background: "rgba(255,255,255,0.015)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10,
          marginBottom: 20
        }}>
          <div style={{
            fontSize: 9,
            letterSpacing: 3,
            color: "#e0e0e0",
            fontFamily: "'JetBrains Mono', monospace",
            marginBottom: 12,
            textTransform: "lowercase"
          }}>
            the full flow
          </div>
          <div style={{
            fontSize: 11,
            color: "#aaa",
            fontFamily: "'JetBrains Mono', monospace",
            lineHeight: 2.4,
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center"
          }}>
            {stages.map((s, i) => (
              <span key={s.id} style={{ display: "inline-flex", alignItems: "center" }}>
                <span
                  style={{
                    color: activeStage === i ? s.color : "#888",
                    cursor: "pointer",
                    transition: "color 0.2s ease",
                    fontSize: 10
                  }}
                  onClick={() => setActiveStage(i)}
                >
                  {s.label}
                </span>
                {i < stages.length - 1 && (
                  <span style={{ color: "#e0e0e0", margin: "0 6px", fontSize: 10 }}>→</span>
                )}
              </span>
            ))}
          </div>
        </div>

        {/* CTA */}
        <div style={{
          padding: "22px 24px",
          background: "rgba(232, 114, 90, 0.06)",
          border: "1px solid rgba(232, 114, 90, 0.15)",
          borderRadius: 10,
          marginBottom: 20,
          textAlign: "center"
        }}>
          <div style={{
            fontSize: 14,
            color: "#e0e0e0",
            lineHeight: 1.7,
            marginBottom: 8
          }}>
            most of us change settings without knowing what we're actually telling the gpu to do. understanding the pipeline doesn't make us engineers ... it makes us better artists
          </div>
          <div style={{
            fontSize: 12,
            color: "#bbb",
            fontFamily: "'JetBrains Mono', monospace"
          }}>
            what should we map next? vae architectures? lora injection? controlnet routing?
          </div>
        </div>

        {/* Footer */}
        <div style={{
          fontSize: 9,
          color: "#777",
          fontFamily: "'JetBrains Mono', monospace",
          textAlign: "center",
          letterSpacing: 3,
          padding: "16px 0",
          textTransform: "lowercase"
        }}>
          og × rtx 5090 × flux dev
        </div>
      </div>
    </div>
  );
}
