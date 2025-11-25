
## Overview of the Project

This project focuses on automatically segmenting sentences into meaningful multiword expressions (MWEs) such as “looked up” or “take off,” which traditional word-by-word processing fails to capture. Existing work on MWE segmentation is limited and does not explore parallelization. This project accelerates MWE detection by parallelizing both the lookup-table construction and the dynamic programming algorithm using CUDA, aiming to significantly reduce the O(n²) runtime of current approaches.

## Project Presentation & Demo

We have prepared a **short recorded presentation and demonstration** of our project. You can view it here: [Watch the Presentation & Demo](https://drive.google.com/file/d/1cGnBCg4dX4wOz6MsdRu4y6Ue9djnsRNv/view?usp=sharing)


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
For a sentence like **"she looked up and down the hallway"**, multiple segmentation candidates are possible. The correct final segmentation is the one containing the **minimum number of valid MWEs**, as shown in the comparison table (see below).

| Candidate Output                                                              | # of MWEs |
|-------------------------------------------------------------------------------|--------------|
| ⟨she⟩ ⟨looked⟩ ⟨up⟩ ⟨and⟩ ⟨down⟩ ⟨the⟩ ⟨hallway⟩                      | 7            |
| ⟨she⟩ ⟨looked up⟩ ⟨and⟩ ⟨down⟩ ⟨the⟩ ⟨hallway⟩                      | 6            |
| ⟨she⟩ ⟨looked⟩ ⟨up and down⟩ ⟨the⟩ ⟨hallway⟩                      | 5            |
| **⟨she⟩ ⟨looked up and down⟩ ⟨the⟩ ⟨hallway⟩**                            | **4**            |


### Valid MWEs
To determine whether a sequence of words forms a meaningful unit, the system uses a dictionary of English multiword expressions. This dictionary is used to build a lookup function that takes any sequence of words and returns a Boolean value indicating whether it is a valid MWE.

### Summary of Inputs and Outputs

**Program Inputs**
- A dataset `X` of input sentences.  
- A dictionary used to build a lookup table of valid MWEs.  

**Program Output**
- A dataset `Y` of segmented sentences, where each sentence is segmented using the smallest valid set of MWEs.

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

...the following DP table can be constructed:

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

2. **Version 2 (Consultation Implementation)**  
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

## Evaluation Methodology

To properly evaluate our various baseline sequential and CUDA-parallelized implementations described previously, we tested our algorithms using real data. Two main sources of data were required: the MWE dictionary for lookup table creation, and a set of test sentences containing MWEs.

### 1. MWE Dictionary
To obtain an established source of MWEs, we used **WordNet** (Miller, 1994), a lexical database of English words that organizes them into sets of synonyms (synsets) and captures semantic relationships between them. WordNet has been widely used in computational linguistics for tasks such as word sense disambiguation and semantic analysis.

After extracting MWEs from WordNet, we performed preprocessing:
- For nouns, we generated plural forms (e.g., "light bulb" → "light bulbs").  
- For verbs, we generated all common inflections:  
  - Base form (e.g., "lighten up")  
  - Past tense (e.g., "lightened up")  
  - Present participle (e.g., "lightening up")  
  - Past participle (e.g., "lightened up")  
  - Third-person singular present (e.g., "lightens up")


After preprocessing, the MWEs were exported as a text file `mwe.txt`, which was imported by our main program implementations as the MWE dictionary. The code used to create this MWE dictionary can be found in the notebook **`MWE-Generator.ipynb`**.


### 2. Test Sentences
We collected test sentences from two sources:  
1. **Kaggle Dataset:** ["Random English Sentences" by NikiTricky](https://www.kaggle.com/datasets/nikitricky/random-english-sentences).
2. **Created Sentences:** Constructed from scratch to include MWEs.

We selected a total of **100 sentences**, divided as follows:  
- **Short Sentences:** 50 sentences with 10-20 words. 
- **Long Sentences:** 50 sentences with more than 20 words.

These sentences were used consistently across all implementations to evaluate execution speed. In this light, we evaluated our implementations under three different scenarios:  
1. Using only the 50 short sentences.
2. Using only the 50 long sentences.
3. Using all 100 sentences (short + long). 

## Performance Results: Execution Time and Speedup Factors

The tables below show the performance of each implementation. We conducted 10 runs for each approach and calculated the average execution time. For GPU implementations, the reported times include not only the kernel execution time but also memory transfer overheads, specifically device-to-host (DtoH) and host-to-device (HtoD) transfers. 

Following this, the GPU times in the tables are presented in the format: **Kernel Time + DtoH Time + HtoD Time = Overall Time**


### Lookup Table Performance (50 Short Sentences)

| Run # | CPU Lookup | GPU Lookup (v1) | GPU Lookup (v2) |
|-------|-----------|-----------------|----------------|
| 1     | 15.203 ms | 0.16249 + 0.10509 + 75.346 = 75.61358 ms | 0.15798 + 0.09264 + 1.2733 = 1.52392 ms |
| 2     | 17.429 ms | 0.17379 + 0.11274 + 77.693 = 77.97953 ms | 0.15802 + 0.1065 + 1.1989 = 1.46342 ms |
| 3     | 18.079 ms | 0.15667 + 0.10182 + 70.007 = 70.26549 ms | 0.15946 + 0.11357 + 1.5457 = 1.81873 ms |
| 4     | 16.038 ms | 0.17155 + 0.11552 + 73.519 = 73.80607 ms | 0.15808 + 0.083869 + 1.1027 = 1.344649 ms |
| 5     | 17.618 ms | 0.16787 + 0.10006 + 77.635 = 77.90293 ms | 0.16096 + 0.091645 + 1.1145 = 1.367105 ms |
| 6     | 11.243 ms | 0.16240 + 0.097917 + 66.710 = 66.970317 ms | 0.15798 + 0.08835 + 0.9934 = 1.23973 ms |
| 7     | 17.292 ms | 0.16445 + 0.10067 + 59.955 = 60.22012 ms | 0.15766 + 0.079934 + 0.88146 = 1.119054 ms |
| 8     | 18.892 ms | 0.15750 + 0.10272 + 65.534 = 65.79422 ms | 0.16794 + 0.08707 + 1.2544 = 1.50941 ms |
| 9     | 17.138 ms | 0.16147 + 0.10426 + 64.531 = 64.79673 ms | 0.15792 + 0.084 + 1.2264 = 1.46832 ms |
| 10    | 10.688 ms | 0.16541 + 0.10669 + 60.774 = 61.0461 ms | 0.15789 + 0.083904 + 1.0815 = 1.323294 ms |
| **AVG** | 15.962 ms | 69.43951 ms | 1.417763 ms |
| **Speedup vs CPU** | 1× | 0.23× | 11.26× |

### Lookup Table Performance (50 Long Sentences)

| Run # | CPU Lookup | GPU Lookup v1 | GPU Lookup v2 |
|-------|-----------|---------------|---------------|
| 1     | 48.686 ms | 0.16617 + 0.10745 + 60.333 = 60.60662 ms | 0.17104 + 0.092384 + 1.1328 = 1.396224 ms |
| 2     | 40.532 ms | 0.17830 + 0.10893 + 91.326 = 91.61323 ms | 0.16800 + 0.087103 + 1.0122 = 1.267303 ms |
| 3     | 51.607 ms | 0.17664 + 0.10518 + 60.079 = 60.36082 ms | 0.16893 + 0.086112 + 1.2977 = 1.552742 ms |
| 4     | 34.072 ms | 0.16538 + 0.11014 + 96.286 = 96.56152 ms | 0.16848 + 0.094591 + 1.3486 = 1.611671 ms |
| 5     | 36.377 ms | 0.17142 + 0.10739 + 78.285 = 78.56381 ms | 0.16787 + 0.10256 + 1.4635 = 1.73393 ms |
| 6     | 49.610 ms | 0.17334 + 0.11002 + 71.981 = 72.26436 ms | 0.16819 + 0.10195 + 1.4377 = 1.70784 ms |
| 7     | 49.918 ms | 0.17152 + 0.10333 + 60.006 = 60.28085 ms | 0.16848 + 0.091199 + 0.95628 = 1.215959 ms |
| 8     | 38.602 ms | 0.16534 + 0.11302 + 86.721 = 86.99936 ms | 0.16877 + 0.10358 + 1.1530 = 1.42535 ms |
| 9     | 49.323 ms | 0.16669 + 0.10742 + 80.675 = 80.94911 ms | 0.16925 + 0.088480 + 1.2543 = 1.51203 ms |
| 10    | 51.378 ms | 0.16582 + 0.10515 + 91.152 = 91.42297 ms | 0.16848 + 0.097183 + 1.2024 = 1.468063 ms |
| **AVG** | 45.0105 ms | 77.96227 ms | 1.489111 ms |
| **Speedup vs CPU** | 1× | 0.58× | 30× |

### Lookup Table Performance (100 Sentences: Short + Long)

| Run # | CPU Lookup | GPU Lookup | GPU Lookup v2 |
|-------|-----------|------------|---------------|
| 1     | 72.239 ms | 0.32730 + 0.19035 + 122.36 = 122.87765 ms | 0.32278 + 0.18470 + 1.0727 = 1.58018 ms |
| 2     | 60.943 ms | 0.32238 + 0.19494 + 116.55 = 117.06732 ms | 0.32310 + 0.17859 + 1.1717 = 1.67339 ms |
| 3     | 72.287 ms | 0.33265 + 0.19877 + 122.38 = 122.91142 ms | 0.32409 + 0.17295 + 1.0895 = 1.58654 ms |
| 4     | 65.595 ms | 0.33205 + 0.19544 + 125.95 = 126.47749 ms | 0.32288 + 0.17603 + 1.0334 = 1.53231 ms |
| 5     | 68.312 ms | 0.34808 + 0.21349 + 139.20 = 139.76157 ms | 0.32268 + 0.18105 + 1.1449 = 1.64863 ms |
| 6     | 64.234 ms | 0.29570 + 0.18220 + 130.44 = 130.9179 ms  | 0.32326 + 0.17545 + 1.1282 = 1.62691 ms |
| 7     | 60.586 ms | 0.32741 + 0.19740 + 134.85 = 135.37481 ms | 0.32297 + 0.18371 + 1.1641 = 1.67078 ms |
| 8     | 65.549 ms | 0.33573 + 0.21414 + 138.54 = 139.08987 ms | 0.32611 + 0.19878 + 1.3570 = 1.88189 ms |
| 9     | 60.790 ms | 0.30312 + 0.18390 + 121.55 = 122.03702 ms | 0.32607 + 0.19005 + 1.3535 = 1.86962 ms |
| 10    | 43.954 ms | 0.32494 + 0.18943 + 110.93 = 111.44437 ms | 0.33436 + 0.18067 + 1.4055 = 1.92053 ms |
| **AVG** | 63.449 ms | 126.796 ms | 1.699 ms |
| **Speedup vs CPU** | 1x | 0.50x | 37.34x |


### DP MWE-Minimization Performance (50 Short Sentences)

| Run # | CPU DP | GPU DP v1 | GPU DP v2 | GPU DP v3 | GPU DP v4 |
|-------|--------|-----------|-----------|-----------|-----------|
| 1     | 3.002 ms | 1.6881 + 0.10592 + 0.096031 = 1.890051 ms | 0.44197 + 0.079489 + 0.095971 = 0.61743 ms | 0.56089 + 0.095646 + 0.093693 = 0.750229 ms | 0.47891 + 0.08115 + 0.09379 = 0.65385 ms |
| 2     | 2.333 ms | 1.6697 + 0.10583 + 0.095072 = 1.870602 ms | 0.44536 + 0.087618 + 0.095969 = 0.628947 ms | 0.56082 + 0.090592 + 0.093952 = 0.745364 ms | 0.47856 + 0.080256 + 0.095008 = 0.653824 ms |
| 3     | 1.455 ms | 1.6684 + 0.10000 + 0.096384 = 1.864784 ms | 0.44389 + 0.092387 + 0.095106 = 0.631383 ms | 0.56093 + 0.10396 + 0.093855 = 0.758745 ms | 0.47827 + 0.087359 + 0.094014 = 0.659643 ms |
| 4     | 2.162 ms | 1.6986 + 0.10144 + 0.095296 = 1.895336 ms | 0.44186 + 0.085761 + 0.095809 = 0.62343 ms | 0.56111 + 0.094461 + 0.094016 = 0.749587 ms | 0.47859 + 0.080509 + 0.093884 = 0.652983 ms |
| 5     | 2.055 ms | 1.6967 + 0.10992 + 0.095009 = 1.901629 ms | 0.44216 + 0.09325 + 0.095299 = 0.630709 ms | 0.56131 + 0.10474 + 0.094847 = 0.760897 ms | 0.47907 + 0.10246 + 0.094624 = 0.676154 ms |
| 6     | 2.120 ms | 1.6490 + 0.096929 + 0.095008 = 1.840937 ms | 0.44193 + 0.10295 + 0.09485 = 0.63973 ms | 0.56169 + 0.10237 + 0.094846 = 0.758906 ms | 0.47820 + 0.085055 + 0.093984 = 0.657239 ms |
| 7     | 2.256 ms | 1.7018 + 0.10627 + 0.095969 = 1.904039 ms | 0.44200 + 0.10119 + 0.094563 = 0.637753 ms | 0.56176 + 0.098847 + 0.094909 = 0.755516 ms | 0.47849 + 0.10726 + 0.094848 = 0.680598 ms |
| 8     | 1.568 ms | 1.6658 + 0.095841 + 0.09456 = 1.856201 ms | 0.44174 + 0.10154 + 0.095937 = 0.639217 ms | 0.56121 + 0.099071 + 0.093984 = 0.754265 ms | 0.47808 + 0.091136 + 0.093854 = 0.66307 ms |
| 9     | 2.243 ms | 1.7011 + 0.10688 + 0.095104 = 1.903084 ms | 0.44308 + 0.11242 + 0.094854 = 0.650354 ms | 0.56108 + 0.09376 + 0.09408 = 0.74892 ms | 0.47840 + 0.09987 + 0.093727 = 0.671997 ms |
| 10    | 2.169 ms | 1.6797 + 0.10218 + 0.095040 = 1.87692 ms | 0.44257 + 0.11415 + 0.096002 = 0.652722 ms | 0.56121 + 0.084927 + 0.095007 = 0.741144 ms | 0.47811 + 0.10314 + 0.094847 = 0.676097 ms |
| AVG   | 2.1363 ms | 1.880358 ms | 0.635168 ms | 0.752357 ms | 0.664546 ms |
| **Speedup vs CPU** | 1x | 1.14x | 3.36x | 2.84x | 3.21x |


### DP MWE-Minimization Performance (50 Long Sentences)

| Run # | CPU DP | GPU DP v1 | GPU DP v2 | GPU DP v3 | GPU DP v4 |
|-------|--------|-----------|-----------|-----------|-----------|
| 1     | 7.453 ms | 3.0010 + 0.10425 + 0.10643 = 3.21168 ms | 0.78276 + 0.098302 + 0.10560 = 0.986662 ms | 1.0297 + 0.12685 + 0.10627 = 1.26282 ms | 0.86278 + 0.10989 + 0.10550 = 1.07817 ms |
| 2     | 7.531 ms | 2.9877 + 0.10054 + 0.10669 = 3.19493 ms | 0.78375 + 0.092894 + 0.10682 = 0.983464 ms | 1.0292 + 0.12464 + 0.10675 = 1.26059 ms | 0.86012 + 0.10365 + 0.10538 = 1.06915 ms |
| 3     | 7.012 ms | 2.9979 + 0.10416 + 0.10678 = 3.20884 ms | 0.78318 + 0.11706 + 0.10573 = 1.00597 ms | 1.0260 + 0.099743 + 0.10672 = 1.23246 ms | 0.86224 + 0.11033 + 0.10650 = 1.07907 ms |
| 4     | 7.756 ms | 3.0202 + 0.10086 + 0.10592 = 3.22698 ms | 0.78312 + 0.099262 + 0.10662 = 0.989002 ms | 1.0300 + 0.13382 + 0.10653 = 1.27035 ms | 0.86275 + 0.10746 + 0.10678 = 1.07699 ms |
| 5     | 3.997 ms | 2.9891 + 0.096576 + 0.10563 = 3.19131 ms | 0.78330 + 0.11264 + 0.10538 = 1.00132 ms | 1.0309 + 0.10442 + 0.10528 = 1.24060 ms | 0.85958 + 0.10432 + 0.10643 = 1.07033 ms |
| 6     | 7.730 ms | 3.0056 + 0.10915 + 0.10669 = 3.22144 ms | 0.77944 + 0.11433 + 0.10669 = 1.00046 ms | 1.0298 + 0.11530 + 0.10640 = 1.25150 ms | 0.85980 + 0.10950 + 0.10653 = 1.07583 ms |
| 7     | 7.802 ms | 2.9820 + 0.10125 + 0.10627 = 3.18952 ms | 0.77966 + 0.11283 + 0.10650 = 0.99899 ms | 1.0263 + 0.11491 + 0.10678 = 1.24799 ms | 0.86224 + 0.11110 + 0.10557 = 1.07891 ms |
| 8     | 8.538 ms | 2.9916 + 0.091264 + 0.10646 = 3.18932 ms | 0.78315 + 0.10438 + 0.10582 = 0.99335 ms | 1.0298 + 0.11219 + 0.10595 = 1.24794 ms | 0.85939 + 0.12048 + 0.10675 = 1.08662 ms |
| 9     | 7.599 ms | 2.9894 + 0.10534 + 0.10650 = 3.20124 ms | 0.75296 + 0.11744 + 0.10570 = 0.97610 ms | 1.0261 + 0.11334 + 0.10659 = 1.24603 ms | 0.86332 + 0.10979 + 0.10630 = 1.07941 ms |
| 10    | 6.882 ms | 2.9958 + 0.10090 + 0.10652 = 3.20322 ms | 0.75296 + 0.10886 + 0.10579 = 0.96761 ms | 1.0306 + 0.12118 + 0.10659 = 1.25837 ms | 0.86012 + 0.11450 + 0.10650 = 1.08112 ms |
| **AVG** | 7.23 ms | 3.20385 ms | 0.99029 ms | 1.25187 ms | 1.07756 ms |
| **Speedup vs CPU** | 1x | 2.26x | 7.30x |  5.78x | 6.71x

### DP MWE-Minimization Performance (100 Sentences: Short + Long)

| Run # | CPU DP | GPU DP v1 | GPU DP v2 | GPU DP v3 | GPU DP v4 |
|-------|--------|-----------|-----------|-----------|-----------|
| 1     | 10.580 ms | 4.6624 + 0.20358 + 0.20131 = 5.06729 ms | 1.2225 + 0.18893 + 0.20166 = 1.61309 ms | 1.5833 + 0.19513 + 0.19776 = 1.97619 ms | 1.3345 + 0.17398 + 0.19981 = 1.70829 ms |
| 2     | 9.522 ms | 4.6654 + 0.21597 + 0.20071 = 5.08208 ms | 1.2173 + 0.20067 + 0.19978 = 1.61775 ms | 1.5830 + 0.21472 + 0.19843 = 1.99615 ms | 1.3350 + 0.17203 + 0.19811 = 1.70514 ms |
| 3     | 8.988 ms | 4.6511 + 0.19229 + 0.20019 = 5.04358 ms | 1.2174 + 0.21728 + 0.20051 = 1.63519 ms | 1.5827 + 0.21568 + 0.19936 = 1.99774 ms | 1.3347 + 0.17405 + 0.19949 = 1.70824 ms |
| 4     | 10.025 ms | 4.6555 + 0.19728 + 0.20288 = 5.05566 ms | 1.2179 + 0.20637 + 0.20272 = 1.62699 ms | 1.5827 + 0.20214 + 0.19773 = 1.98257 ms | 1.3346 + 0.17238 + 0.19805 = 1.70503 ms |
| 5     | 5.541 ms | 4.6480 + 0.18739 + 0.20208 = 5.03747 ms | 1.2179 + 0.19927 + 0.20096 = 1.61813 ms | 1.5830 + 0.18672 + 0.19769 = 1.96741 ms | 1.3342 + 0.16931 + 0.19990 = 1.70341 ms |
| 6     | 5.883 ms | 4.6465 + 0.19219 + 0.20100 = 5.03969 ms | 1.2197 + 0.19437 + 0.20016 = 1.61423 ms | 1.5828 + 0.19530 + 0.19750 = 1.9756 ms | 1.3345 + 0.18976 + 0.19974 = 1.724 ms |
| 7     | 9.856 ms | 4.6552 + 0.19107 + 0.20196 = 5.04823 ms | 1.2288 + 0.21341 + 0.20199 = 1.6442 ms | 1.5832 + 0.20493 + 0.20025 = 1.98838 ms | 1.3340 + 0.18704 + 0.19994 = 1.72098 ms |
| 8     | 5.868 ms | 4.6784 + 0.20631 + 0.20125 = 5.08596 ms | 1.2289 + 0.22013 + 0.20080 = 1.64983 ms | 1.5835 + 0.22729 + 0.19987 = 2.01066 ms | 1.3346 + 0.18784 + 0.19971 = 1.72215 ms |
| 9     | 10.725 ms | 4.6723 + 0.20007 + 0.20048 = 5.07285 ms | 1.2287 + 0.22928 + 0.20000 = 1.65798 ms | 1.5834 + 0.19328 + 0.20013 = 1.97681 ms | 1.3343 + 0.18470 + 0.19949 = 1.71849 ms |
| 10    | 10.180 ms | 4.6624 + 0.20308 + 0.20250 = 5.06798 ms | 1.2317 + 0.22426 + 0.20374 = 1.6597 ms | 1.5831 + 0.19542 + 0.19926 = 1.97778 ms | 1.3346 + 0.17251 + 0.20048 = 1.70759 ms |
| AVG   | 8.7168 ms | 5.060079 ms | 1.633709 ms |  1.984929 ms | 1.712332 ms |
| **Speedup vs CPU** | 1x | 1.72x | 5.34x | 4.39x | 5.09x |

## Discussion of Results

### Discussion on Lookup Implementation

For the lookup function, the CPU kernel initially outperformed the GPU kernel (GPU Lookup v1). This is not because the CPU is inherently faster; rather, each time we launch the GPU kernel for a new sentence, the entire dictionary lookup must be transferred from the host (CPU) to the device (GPU). 

Analysis shows that most of the GPU execution time is **spent in this HtoD (host-to-device) memory transfer**. These heavy transfers caused GPU Lookup v1 to sometimes run only half as fast as the CPU (0.58× for long sentences and 0.50× for the merged set) or even a quarter as fast (0.23× for short sentences).

To address this, we created a **second version of the GPU kernel (GPU Lookup v2)** that keeps the dictionary in GPU memory across multiple sentence kernels, transferring it only once. With this optimization, the speedup compared to the CPU kernel improves dramatically: for short sentences, GPU Lookup v2 achieves approximately **11.3×** speedup; for long sentences, around **30×**; and for the merged 100-sentence set, about **37.34×**.

Overall, this exercise in optimizing overhead only serves to exemplify the importance of carefully examining the overhead involved in GPU implementations. Without breaking down where time is actually spent, performance comparisons between CPU and GPU kernels can be misleading. More importantly, analyzing overhead such as memory transfers provides actionable insights that guide how GPU implementations can be improved moving forward.


### Discussion on MWE-Minimization DP

For the **MWE minimization algorithm**, we hypothesized that shorter sentences would have a smaller speedup factor compared to the CPU version. This is because the DP table for short sentences has smaller diagonals, resulting in fewer cells processed in parallel.  

In this light, our results support this hypothesis:  
- **Short sentences:** 3.36× speedup using GPU DP v2 over CPU DP  
- **Long sentences:** 7.30× speedup using GPU DP v2 over CPU DP  
- **Merged set:** 5.34× speedup using GPU DP v2 over CPU DP  

Furthermore, across all test sets, the **GPU v2 implementation consistently outperforms GPU v3 and v4**, showing that non-coalesced memory is not a significant issue for the diagonal-based DP implementation. GPU v2 represents the best-performing GPU optimization and is used for speedup comparisons. Following this, we observe a consistent performance trend across all datasets: **v2 > v4 > v3**. After a deeper analysis of the DP algorithm and its memory-access behavior, we theorize that this trend is still related to memory non-coalescence—but at a more subtle level.

Across versions v2, v3, and v4, what we changed was the *order in which the DP algorithm fills the table*—that is, the outer loop traversal pattern. However, closer inspection of the algorithm shows that each DP cell computation depends heavily on an **inner loop**, and this inner loop executes far more times than the outer loop. Because of this, the inner loop’s memory-access pattern has a much stronger effect on total performance.

When examining how the inner loop accesses memory in each version, we found the following:

1. **v2**: The inner loop accesses DP cells in **row-major (fully contiguous)** and **column-major (semi-contiguous)** patterns.  
2. **v3**: The inner loop accesses DP cells **diagonally (non-contiguous)** and **column-major (semi-contiguous)**.  
3. **v4**: The inner loop accesses DP cells **diagonally (non-contiguous)** and **row-major (fully contiguous)**.

See the figures below for visualizations of these access patterns.

**Figure 1.** GPU DP v2 solving for the final cell labeled **X**.  
Cells that need to be accessed as dependencies are labeled **O**, while unused cells are labeled **–**.  
Dependency cells are accessed in **row-major** and **column-major** order.

|     | 0   | 1   | 2   | 3   |
|-----|-----|-----|-----|-----|
| 0   | O   | O   | O   | X   |
| 1   |     | –   | –   | O   |
| 2   |     |     | –   | O   |
| 3   |     |     |     | O   |


**Figure 2.** GPU DP v3 solving for the final cell labeled **X**.  
Cells that need to be accessed as dependencies are labeled **O**, while unused cells are labeled **–**.  
Dependency cells are accessed in **diagonal** and **column-major** order.

|     | 0   | 1   | 2   | 3   |
|-----|-----|-----|-----|-----|
| 0   | O   | –   | –   | O   |
| 1   | O   | –   | O   |     |
| 2   | O   | O   |     |     |
| 3   | X   |     |     |     |

**Figure 3.** GPU DP v4 solving for the final cell labeled **X**.  
Cells that need to be accessed as dependencies are labeled **O**, while unused cells are labeled **–**.  
Dependency cells are accessed in **row-major** and **diagonal** order.

|     | 0   | 1   | 2   | 3   |
|-----|-----|-----|-----|-----|
| 0   | O   | O   | O   | X   |
| 1   | –   | –   | O   |     |
| 2   | –   | O   |     |     |
| 3   | O   |     |     |     |

Given this, the performance trend becomes clear: versions whose inner loops access memory more contiguously perform better overall. Row-major access is the most efficient, as memory is fully contiguous. Column-major access is moderately efficient—better than diagonal access—because the "jump" between consecutive memory locations is smaller, reducing cache misses. Diagonal access is consistently the worst due to its long, irregular memory strides.

This explains why **v2** performs best: although its *outer loop* traverses the DP table diagonally, this arrangement allows its *inner loop* to operate with the most contiguous and cache-friendly memory accesses. This highlights the importance of carefully considering memory-access patterns—especially contiguity—when designing algorithms for GPUs and CUDA.


## Screenshots of Program Execution

### CPU: Lookup Kernel

<img width="776" height="311" alt="image" src="https://github.com/user-attachments/assets/7b6b264e-0d9e-4f09-a2b3-f28812b4fa04" />

### GPU: Lookup Kernel (v1)

<img width="780" height="318" alt="image" src="https://github.com/user-attachments/assets/84f6d584-29d8-47bf-8882-330e402f8567" />
<img width="1248" height="309" alt="image" src="https://github.com/user-attachments/assets/ab16eab8-ca46-4926-8eb6-6893238dca32" />

### GPU: Lookup Kernel (v2)

<img width="1015" height="390" alt="image" src="https://github.com/user-attachments/assets/8dbed125-98d2-4453-a846-4372d219d7a5" />
<img width="1523" height="383" alt="image" src="https://github.com/user-attachments/assets/c4af4175-70f1-4b33-b9b3-f00f4aeebc52" />

### CPU: MWE-Minimization DP

<img width="778" height="305" alt="image" src="https://github.com/user-attachments/assets/84e2ec70-9f4f-44b4-93aa-f2da8ca79ff2" />

### GPU: MWE Minimization DP (v1)

<img width="782" height="274" alt="image" src="https://github.com/user-attachments/assets/3ad7b897-5428-470f-8b08-f3d54c20ba75" />
<img width="915" height="331" alt="image" src="https://github.com/user-attachments/assets/ae7f1120-a945-4f53-9300-015364118c4d" />

### GPU: MWE Minimization DP (v2)

<img width="770" height="312" alt="image" src="https://github.com/user-attachments/assets/06f8fe0d-4028-4912-9a41-e89f7088163f" />
<img width="874" height="326" alt="image" src="https://github.com/user-attachments/assets/601cb531-6fe3-45e4-ab3f-1daae05289be" />

### GPU: MWE Minimization DP (v3)

<img width="774" height="310" alt="image" src="https://github.com/user-attachments/assets/6d1c59cf-3030-4316-b7ec-91191a0d386e" />
<img width="835" height="327" alt="image" src="https://github.com/user-attachments/assets/2e608385-7881-4ed2-9875-27f1d4932a7e" />

### GPU: MWE Minimization DP (v4)

<img width="772" height="308" alt="image" src="https://github.com/user-attachments/assets/f0158b26-7efc-41aa-8f43-b9d1bb155d72" />
<img width="842" height="328" alt="image" src="https://github.com/user-attachments/assets/21ec10f8-f710-4c19-a4bb-fc16f912e2e1" />


## Conclusion

This work has demonstrated the application of **CUDA parallelization** to the problem of segmenting sentences into **MWEs**. By presenting various **implementations** and **iterations**, we provided insights into **minimizing overhead**, particularly in the context of **CPU-GPU communication**, and highlighted the importance of designing algorithms that take advantage of **coalesced memory access**. Furthermore, given that our CUDA implementations achieved significant **speedups** compared to traditional **sequential implementations**, this project exemplifies the power of **parallelization** and the substantial **performance gains** that can be achieved through careful and thoughtful **GPU optimization**.

## Use of AI Assistance

For transparency, **ChatGPT** was used to assist in **grammar refinement**, **wording improvements**, and **formatting** throughout this report. Prompts were primarily of the form: *“[Paragraph]. Please fix the wording of this section.”* All **analysis, insights, interpretations, and technical discussions** originate solely from the authors and reflect their own understanding and reasoning.  

All AI-assisted outputs were **carefully reviewed and verified by the authors** to ensure that they were **factually correct** and **null of hallucinations**.

## Authors of This Project
- Encinas, Robert Joachim Olivera
- Lim, Lanz Kendall Yong
- Tan, Tyler Justin Hernandez

## References

- Constant, M., Eryiğit, G., Monti, J., Van Der Plas, L., Ramisch, C., Rosner, M., & Todirascu-Courtier, A. (2017). *Survey: Multiword Expression Processing: A Survey*. Computational Linguistics, 43, 837–892. [https://doi.org/10.1162/coli_a_00302](https://doi.org/10.1162/coli_a_00302)

- Kanclerz, K., & Piasecki, M. (2022). *Deep Neural Representations for Multiword Expressions Detection*, 444–453. [https://doi.org/10.18653/v1/2022.acl-srw.36](https://doi.org/10.18653/v1/2022.acl-srw.36)

- Miller, G. A. (1994). *WordNet: A Lexical Database for English*. In *Human Language Technology: Proceedings of a Workshop held at Plainsboro, New Jersey, March 8–11, 1994*. [https://aclanthology.org/H94-1111/](https://aclanthology.org/H94-1111/)

- Schneider, N., Danchik, E., Dyer, C., & Smith, N. (2014). *Discriminative Lexical Semantic Segmentation with Gaps: Running the MWE Gamut*. Transactions of the Association for Computational Linguistics, 2, 193–206. [https://doi.org/10.1162/tacl_a_00176](https://doi.org/10.1162/tacl_a_00176)

- Williams, J. (2016). *Boundary-based MWE segmentation with text partitioning*, 1–10. [https://doi.org/10.18653/v1/w17-4401](https://doi.org/10.18653/v1/w17-4401)
