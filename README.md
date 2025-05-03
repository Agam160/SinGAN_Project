# Self-Supervised Image Segmentation from a Single Image

### Problem Definition

This project addresses the problem of image segmentation without any ground truth labels, by using a self-supervised learning approach. The input is a single RGB image, and the output is a segmentation map where each pixel is assigned a region label. 

This task is important because obtaining labelled segmentation data is generally labour intensive and expensive. Furthermore, self-supervised methods learn from the image itself, making them suitable for scenarios where manual annotations are not available.

A successful outcome involves generating a visually semantically meaningful segmentation map, which can be evaluated qualitatively by observing the continuity of object regions and by examining cluster separability.

- **Input image**: \( I \in \mathbb{R}^{H \times W \times 3} \)  
- **Segmentation map**: \( S \in \mathbb{R}^{H \times W} \)

---

### Method

The code segments an image into meaningful regions by learning how different parts of the image relate to each other using **contrastive self-supervised learning**:

1. **Patch Sampling**:  
   - Random patches of size 32×32 are sampled from the input image.
   - Each patch is augmented twice (cropping, color jittering, flipping) to create training pairs.

2. **CNN Encoder**:  
   - A small CNN encodes each patch into a low-dimensional embedding.
   - Architecture:  
     - 2 convolutional layers  
     - ReLU activations  
     - Adaptive average pooling

3. **Contrastive Loss - NT-Xent**:  
   - Encourages similar patches to have close embeddings, and dissimilar ones to be far apart.

   ```math
   l(i, j) = -\log \left( \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp(\text{sim}(z_i, z_k)/\tau)} \right)
   ```
4. **Segmentation via Clustering:**
    - Overlapping patches are extracted from the full image using a sliding window.
    - The trained encoder generates embeddings for these patches.
    - K-Means clustering is applied to group the embeddings into region labels.
    - The final segmentation map is reconstructed by averaging cluster assignments in overlapping regions.
## Results

The experiments were conducted on a single image of a building complex, with 500 random patches used for training. The encoder was trained for 50 epochs using the Adam optimizer and a learning rate of 1e-3. After training, 32x32 patches were extracted every 16 pixels to create overlapping embeddings for segmentation. The CNN encoder used in this work is deliberately simple, comprising two convolutional layers and pooling, chosen to reduce overfitting and ensure computational efficiency. However, this simplicity may limit the capacity to capture higher-level semantics or texture patterns, especially in more complex scenes. Similarly, the choice of 32×32 patch size represents a trade-off, large enough to contain contextual cues, but small enough to provide detailed spatial information.

As shown in Figure 1 training over 50 epochs revealed a steady decline in the NT-Xent loss, indicating that the encoder progressively learned more discriminative representations. However, the rate of improvement diminished toward the later epochs, suggesting a point of diminishing returns. This highlights the trade-off between compute time and representation quality, especially in single-image training scenarios

**Quantitative Evaluation:**
Although no ground truth is available for numerical evaluation, clustering quality was indirectly assessed through visual coherence and cluster compactness. For example, using 10 clusters resulted in visually distinct regions corresponding to different parts of the image (e.g., sky, windows, building walls).

**Qualitative Evaluation:**
As shown in Figure 2, the segmented image displays clear separation between structural components of the scene. Regions such as the roof, sky, and glass panes form coherent clusters, indicating that the encoder captured semantically meaningful features despite having no labels.

Furthermore, to explore the effect of cluster granularity on segmentation quality, multiple test cases were evaluated using different numbers of clusters: 3, 5, 7, and 10 as shown in Figure 3. As the number of clusters increased, the segmentation maps revealed progressively finer details. While fewer clusters (e.g., 3 or 5) grouped large regions effectively, they failed to distinguish smaller visual components. In contrast, using 10 clusters produced the most visually coherent and semantically rich segmentation, successfully isolating features like windows, roofs, and sky regions. This suggests that higher cluster counts can better capture structural variation in the image, though overly high values may risk over segmentation.

## Reflection

This work highlights the potential of self-supervised methods for segmentation tasks in low-data regimes. It could benefit areas like medical imaging or satellite imagery, where manual labeling is costly or infeasible. However, the approach may be less effective on cluttered or highly textured scenes, and there is a risk of amplifying data biases if the encoder overfits to low-level patterns. 

## Conclusion

This project demonstrated that self-supervised contrastive learning can effectively segment a single image into semantically meaningful regions. Future improvements could include multi-scale training, spatial consistency constraints, or extension to video inputs for temporal coherence.
