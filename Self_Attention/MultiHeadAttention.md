The goal here is to understand how multi-head attention is implemented step by step and the intuition behind each operation.

Core Concept of Multi-Head Attention
Multi-head attention is a key part of the transformer architecture, allowing the model to focus on different positions of the input sequence in parallel. By splitting the embedding space into multiple "heads," the model can attend to different pieces of information across the sequence. This increases both capacity and robustness of the attention mechanism.

The Code
```python

class MultiHeadAttention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.nh = num_heads
        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: Tensor):
        B, S, C = x.shape

        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        attn = attn.softmax(dim=-1)

        x = attn @ v

        return self.Wo(x.transpose(1, 2).reshape(B, S, C))
```
Line-by-Line Explanation
<br>
<br>

1.
```python
 __init__(self, hidden_size: int, num_heads: int, bias: bool = True)
```
This is the constructor, and it defines the initialization of the multi-head attention layer.

hidden_size: This is the dimension of the input embeddings (e.g., if your input embeddings are 512-dimensional, hidden_size = 512).
num_heads: The number of attention heads. Each head will process a portion of the hidden size. The hidden size must be divisible by num_heads.
Each attention head operates on a smaller part of the full hidden size (specifically, hidden_size / num_heads).
bias: A boolean flag to determine if bias terms should be included in the linear transformations.
Assertions:
```python
assert hidden_size % num_heads == 0
```
This checks that hidden_size is divisible by num_heads, ensuring that the hidden dimension can be split equally among all heads.

Layers:
```python
self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
```
This is a linear transformation that computes the query, key, and value matrices in one go. It outputs 3 * hidden_size because each element of the input is transformed into three parts: the query (Q), the key (K), and the value (V). The query, key, and value are later split during the forward pass.

```python
self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
```
This is another linear layer that will be used at the end to project the concatenated attention heads back to the original hidden size.

<br>
<br>


2. 
```python
def forward(self, x: Tensor):
```
The forward method defines the computation performed on every input.

```python
B, S, C = x.shape
```
Here, x is a tensor of shape (B, S, C):

B is the batch size.
S is the sequence length (i.e., the number of tokens in the input).
C is the hidden size, or the dimensionality of each token's embedding (e.g., 512).
So, let's say the input x has shape (B=2, S=4, C=8):

B=2: Two sequences (or sentences) in the batch.
S=4: Each sequence has 4 tokens.
C=8: Each token is represented by an 8-dimensional vector.

<br>
<br>

3. 
```python
x = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
```
Here we perform the linear transformation Wqkv on the input tensor x.

self.Wqkv(x) applies the linear transformation to every token in the sequence. Since Wqkv outputs 3 * hidden_size, the output shape is (B, S, C * 3).
For our example where C=8, this would give:

```python
(2, 4, 24)  # (B=2, S=4, C*3=24)
```
This means each token (of size 8) is now represented by a vector of size 24, which contains the concatenated query (Q), key (K), and value (V).

Next, we reshape the output into (B, S, 3, nh, C//nh):

B is the batch size.
S is the sequence length.
3 represents the three components (queries, keys, and values).
nh is the number of attention heads.
C//nh is the size of each head's subspace.
For our example:

nh=2, and C//nh=4. Thus, the new shape will be:
```python
(2, 4, 3, 2, 4)
```
Here:

The 3 corresponds to the queries, keys, and values.
The 2 is the number of attention heads.
The 4 is the dimension of each head's query, key, and value vectors.
<br>
<br>


4. 
```python
q, k, v = x.transpose(3, 1).unbind(dim=2)
```
x.transpose(3, 1) swaps the head dimension (nh) with the sequence length dimension (S). After this, the shape of x becomes (B, nh, S, 3, C//nh):

In our case, the shape will be (2, 2, 4, 3, 4).
.unbind(dim=2) splits the third dimension (which contains the Q, K, and V vectors) into separate tensors for q, k, and v.

At this point:

q, k, and v will each have shape (B, nh, S, C//nh). In our example:
```python
(2, 2, 4, 4)  # (B=2, nh=2, S=4, C//nh=4)
```
This means:

q, k, and v now have 2 attention heads (nh=2), each operating on the 4-dimensional embeddings of 4 tokens in each sequence.
<br>
<br>


5. 
```python
attn = q @ k.transpose(-2, -1)
```
This step computes the attention scores by performing the dot product of the query (q) and the transpose of the key (k).

k.transpose(-2, -1) switches the last two dimensions of k, so its shape becomes (B, nh, C//nh, S). In our example, k will have shape (2, 2, 4, 4) after the transpose.
The dot product q @ k.transpose(-2, -1) calculates the similarity between each query and key, giving a tensor of attention scores with shape (B, nh, S, S).

In our example, this results in:

```python
(2, 2, 4, 4)  # (B=2, nh=2, S=4, S=4)
```
This means that for each of the 2 heads, we now have a 4x4 matrix that represents the attention scores for each token in the sequence (attending to every other token).
<br>
<br>


6. 
```python
attn = attn / math.sqrt(k.size(-1))
```
This normalizes the attention scores by dividing by the square root of the dimensionality of the key vectors (C//nh). This scaling factor prevents the dot products from becoming too large as the dimensionality increases, which stabilizes gradients during training.

In our example, C//nh=4, so the attention scores are scaled by sqrt(4) = 2.
<br>
<br>


7. 
```python
attn = attn.softmax(dim=-1)
```
The softmax function is applied to the attention scores along the last dimension (S) to convert them into probabilities. This ensures that the attention scores for each token sum to 1, effectively representing a weighted distribution over the tokens in the sequence.
<br>
<br>


8. 
```python
x = attn @ v
```
Next, we compute the weighted sum of the value vectors (v) using the attention scores (attn). This is done via matrix multiplication between attn and v, resulting in a tensor of shape (B, nh, S, C//nh).

In our example, the shape will be:

```python
(2, 2, 4, 4)  # (B=2, nh=2, S=4, C//nh=4)
```
This gives the output of each attention head.
<br>
<br>


9. 
```python
return self.Wo(x.transpose(1, 2).reshape(B, S, C))
```
Finally, we:

x.transpose(1, 2) swaps the sequence length and head dimensions, bringing the shape back to (B, S, nh, C//nh).
.reshape(B, S, C) flattens the head dimension, resulting in a tensor of shape (B, S, C).
The linear transformation self.Wo is then applied to project the concatenated attention head outputs back into the original hidden size (C).

Intuition Recap:
Splitting the Embedding Space: Multi-head attention splits the input into multiple smaller embedding spaces, allowing the model to attend to different parts of the sequence in parallel.
Queries, Keys, and Values: These represent different ways of transforming the input. The attention mechanism computes relationships between tokens using dot products of queries and keys.
Attention Scores: The dot product of queries and keys gives a similarity measure, determining how much focus should be placed on each token.
Weighted Sum of Values: The attention scores are used to compute a weighted sum of the value vectors, allowing each token to aggregate information from other tokens.
Final Projection: The concatenated attention outputs from all heads are projected back into the original space.





