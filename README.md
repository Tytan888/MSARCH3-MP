## Overview of the Project

This project focuses on automatically segmenting sentences into meaningful multiword expressions (MWEs) such as “looked up” or “take off,” which traditional word-by-word processing fails to capture. Existing work on MWE segmentation is limited and does not explore parallelization. This project accelerates MWE detection by parallelizing both the lookup-table construction and the dynamic programming algorithm using CUDA, aiming to significantly reduce the O(n²) runtime of current approaches.

## Task: Minimum Multiword Expression (MWE) Segmentation

### Problem Description
Let `X = { W1, W2, ..., Wx }` be a dataset of input sentences.  
Each sentence `W` is a sequence of words represented as:

`W = (w1, w2, ..., wn)`

The goal is to generate an output sentence `V` for each input sentence `W` by grouping certain consecutive words into multiword expressions (MWEs). For example, the words `w1` and `w2` may be combined into a single MWE represented as:

`{ w1, w2 }`

The final output sentence should contain the **fewest possible MWEs** while still preserving the meaning of the original sentence.

Thus, each output sentence takes a form such as:

`V = ( {w1, w2}, ..., wn )`

Across the entire dataset, the resulting outputs form:

`Y = { V1, V2, V3, ..., Vx }`

### Example
For a sentence like **"she looked up and down the hallway"**, multiple segmentation candidates are possible.  
The correct final segmentation is the one containing the **minimum number of valid MWEs**, as shown in the comparison table (see below).

TODO: INSERT TABLE HERE

### Valid MWEs
To determine whether a sequence of words forms a meaningful unit, the system uses a dictionary of English multiword expressions. This dictionary is used to build a lookup function that takes any sequence of words and returns a Boolean value indicating whether it is a valid MWE.

### Summary of Inputs and Outputs

**Program Inputs**
- A dataset `X` of input sentences  
- A dictionary used to build a lookup table of valid MWEs  

**Program Output**
- A dataset `Y` of segmented sentences, where each sentence is rewritten using the smallest valid set of MWEs  

## Algorithm Overview

The algorithm begins by taking a single sentence as input. From this sentence, it generates all possible multiword expression (MWE) candidates by computing every sequential word combination within the sentence.  

If a sentence `W` consists of `n` words, then the total number of candidate phrases can be expressed as:

$$
\sum_{i=1}^{n} i = 1 + 2 + \dots + n = \frac{n(n+1)}{2}
$$

**Example:** 
For the sentence: `"She looked up and down the hallway,"` the set of possible sequential word combinations includes:

<div align="center">

```
["she", "she looked", "she looked up", "she looked up and", "looked", "looked up", # ... and so on]
```
</div>

Once all candidate phrases are enumerated, each phrase is checked against a dictionary of valid MWEs. The dictionary serves as a lookup table that indicates whether a sequence of words forms a recognized multiword expression.  

For example, the phrase `"looked up and down"` is a valid MWE and is retained in the filtered list.

After obtaining the list of valid MWEs, the algorithm identifies the **optimal segmentation** of the sentence — the segmentation that minimizes the total number of MWEs while preserving grammatical and semantic coherence.  

**Example:** For the sentence: `"She looked up and down the hallway,"` the optimal segmentation is:

 
<div align="center">

```
[ "she" ], [ "looked up and down" ], [ "the" ], [ "hallway" ] 
```
</div>
 
This reduces the total count from seven individual words to four MWEs.

This optimization is implemented using a **dynamic programming (DP) approach**. In this setup:  
- The DP table’s rows and columns represent word indices in the sentence.  
- Each cell stores the minimum number of MWEs required to segment that word span.  
- The table is filled iteratively based on substructure dependencies, enabling efficient computation of the global minimum segmentation.

To define the dynamic programming algoirthm more formally, let `lookup(x, y)` be a Boolean function that returns `TRUE` if the span from word `x` to `y` (inclusive) forms a valid MWE. Let `MWE_MIN(x, y)` denote the minimum number of MWEs needed to cover the subsentence from word `x` to word `y`.

The following formula can be used to compute `MWE_MIN(x, y)`:

$$
MWE\_MIN(x, y) =
\begin{cases}
1 & \text{if } x = y \text{ or } lookup(x, y) = \text{TRUE} \\
\min\limits_{x \leq m < y} (MWE\_MIN(x, m) + MWE\_MIN(m + 1, y)) & \text{otherwise}.
\end{cases}
$$

For the sentence "she looked up and down the hallway", and assuming the only MWEs present are:

- "looked up"
- "up and down"
- "down the hallway"
- "looked up and down"

The following DP table can be constructed.

|         | she | looked | up | and | down | the | hallway |
|---------|-----|--------|----|-----|------|-----|---------|
| **she**     | 1   | 2      | 2  | 3   | 2    | 3   | 4       |
| **looked**  |     | 1      | 1  | 2   | 1    | 2   | 3       |
| **up**      |     |        | 1  | 2   | 1    | 2   | 3       |
| **and**     |     |        |    | 1   | 2    | 3   | 2       |
| **down**    |     |        |    |     | 1    | 2   | 1       |
| **the**     |     |        |    |     |      | 1   | 2       |
| **hallway** |     |        |    |     |      |     | 1       |



## Parallelization Strategies

Several stages of the MWE segmentation process can benefit from parallelization. This project primarily utilizes **CUDA** for GPU-based parallel processing, with three main opportunities:

### 1. Sentence-Level Parallelization
Multiple sentences can be processed concurrently, with each CUDA thread handling an entire sentence.  
Each thread constructs its own lookup table and executes the segmentation algorithm independently.  
**Limitation:** This approach requires substantial GPU memory since each thread maintains its own dictionary and dynamic programming table.

### 2. Dictionary Lookup Parallelization
The generation and validation of MWE candidates against the dictionary can be massively parallelized.  
Each GPU thread can verify one or more candidate phrases concurrently, significantly reducing the time needed to build the lookup table.

### 3. Dynamic Programming (DP) Parallelization
The DP segmentation algorithm can be parallelized by exploiting the independence of cells along the same diagonal of the DP matrix.  
- Each cell depends only on values from previous diagonals.  
- All cells on a single diagonal can be computed concurrently, with one thread per cell.  
- This reduces the effective computation time from O(n²) to roughly O(n), aside from minor synchronization overheads.

### Proposed Implementation
For this project, we will implement **dictionary lookup parallelization** and **DP parallelization** together.  
- Sentence-level parallelization is avoided due to excessive memory requirements and complexity.  
- Combining dictionary and cell-level parallelization provides efficient speedup while keeping memory usage manageable and minimizing synchronization overhead in CUDA kernels.

## Related Research / Literature Review (RRL)

Extensive research exists on multiword expressions (MWEs), but most studies focus on **MWE extraction or detection**—identifying valid MWEs from large text corpora (e.g., Constant et al., 2017; Kanclerz & Piasecki, 2022; Schneider et al., 2014).  

In contrast, this project focuses on **MWE segmentation**, where the task is to divide a sentence into meaningful MWE segments given a sentence and a predefined list of valid MWEs (Williams, 2016).

Current literature shows that research on MWE segmentation is limited, and **none have explored parallelizing internal processing steps**, such as lookup table generation or dynamic programming (DP) table computations using CUDA.  

Our proposed implementation differs by **introducing parallelization strategies** for both dictionary lookup and DP segmentation, providing faster execution while retaining semantic coherence. This approach contributes novelty and potential value to the field of computational linguistics and computer science.

## Parallelized MWE Detection and Minimization

The goal of our project is to **speed up the process of detecting multiword expressions (MWEs)** and to **speed up the algorithm for minimizing the number of MWEs in a sentence**. The overall process of our program involves two main stages: lookup table creation and MWE minimization.

### 1. Lookup Table Creation

The first stage requires a predefined list of MWEs, which is stored in the program as a **hashmap**. For each input sentence, all possible contiguous substrings are considered as potential MWEs. Each substring is then checked against the MWE dictionary, and valid matches are marked for later use in the minimization stage.  

Traditionally, this process is performed **sequentially**, where each substring is checked one after another. By **parallelizing** this step, we can evaluate multiple candidate substrings concurrently, as the result of one potential MWE does not depend on another. This allows for significant speedup in building the lookup table.

We developed **two parallelized versions** for lookup table creation:

1. **Version 1 (Milestone 2 Implementation)**  
   - Launches a single GPU kernel per sentence, supplying all necessary data, including the sentence itself and the entire MWE dictionary as a hashmap.  
   - The entire MWE dictionary is sent for every sentence, even though it is static and does not change, resulting in **redundant host-to-device (CPU-to-GPU) memory transfers**.

2. **Version 2**  
   - Similar to Version 1, but the MWE dictionary is sent **only once** and persists in GPU memory.  
   - This reduces redundant memory transfers and **cuts a large portion of host-to-device overhead**, improving overall performance.


### 2. MWE Minimization Using Dynamic Programming

The MWE minimization algorithm uses a **dynamic programming (DP) approach**.  

- Smaller combinations of the sentence are solved first, progressively moving to larger spans.  
- The DP table’s rows and columns represent word indices in the sentence.  
- Each cell stores the **minimum number of MWEs** required to segment that span.  
- The table is filled iteratively based on substructure dependencies, allowing efficient computation of the global minimum segmentation.

We developed four parallelized versions of the MWE minimization algorithm:

1. **Version 1 (Milestone 2 Implementation)**  
   - Diagonal-based parallelization: all cells along a diagonal of the DP table are computed concurrently.  
   - A new GPU kernel is launched for each diagonal, introducing significant kernel launch overhead.  

2. **Version 2**  
   - Launches a single kernel per sentence instead of per diagonal. 
   - Reduces kernel launch overhead signficiantly compared to Version 1. 
   - The diagonal nature of the DP algorithm means that DP cells are computed along diagonals, which **may lead to non-coalesced memory access**, as threads may not necessarily access contiguous memory locations.  

3. **Versions 3 and 4**  
   - Restructure the DP table so that instead of both rows and columns representing start and end word indices, the table’s axes now represent **start word index** and **span length** of the sub-sentence.  
   - This allows the DP cells to be computed in a **row-major or column-major order** (depending on which axis represents start index and which represents span length).  
   - By reorganizing the table this way, memory accesses **may become more coalesced**, potentially improving cache utilization and overall efficiency.
 

#### Notes on Memory Analysis
Initially, we attempted to profile memory behavior using:

```bash
nvcc -lineinfo -G -g yourfile.cu -o app
ncu ./app
```

However, analyzing memory statistics with `ncu` requires root-level access, which was unavailable on the CCS cloud system. In this light, Versions 3 and 4 were developed to explore ways to mitigate potential memory inefficiencies and determine the most effective DP table structure to reduce the impact of non-coalesced access, even without full profiling capabilities.

## Authors of This Project
- Encinas, Robert Joachim Olivera
- Lim, Lanz Kendall Yong
- Tan, Tyler Justin Hernandez

## References

- Constant, M., Eryiğit, G., Monti, J., Van Der Plas, L., Ramisch, C., Rosner, M., & Todirascu-Courtier, A. (2017). *Survey: Multiword Expression Processing: A Survey*. Computational Linguistics, 43, 837–892. [https://doi.org/10.1162/coli_a_00302](https://doi.org/10.1162/coli_a_00302)

- Kanclerz, K., & Piasecki, M. (2022). *Deep Neural Representations for Multiword Expressions Detection*, 444–453. [https://doi.org/10.18653/v1/2022.acl-srw.36](https://doi.org/10.18653/v1/2022.acl-srw.36)

- Schneider, N., Danchik, E., Dyer, C., & Smith, N. (2014). *Discriminative Lexical Semantic Segmentation with Gaps: Running the MWE Gamut*. Transactions of the Association for Computational Linguistics, 2, 193–206. [https://doi.org/10.1162/tacl_a_00176](https://doi.org/10.1162/tacl_a_00176)

- Williams, J. (2016). *Boundary-based MWE segmentation with text partitioning*, 1–10. [https://doi.org/10.18653/v1/w17-4401](https://doi.org/10.18653/v1/w17-4401)
