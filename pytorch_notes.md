
---

# `torch.squeeze()` and `torch.unsqueeze()`

## 1. `torch.squeeze()`

### Purpose

Removes dimensions of size **1**.

### Basic Example

```python
x = torch.randn(4, 1, 3, 1)
print(x.shape)  # torch.Size([4, 1, 3, 1])

y = x.squeeze()
print(y.shape)  # torch.Size([4, 3])
```

All dimensions of size 1 are removed.

---

### Squeeze Specific Dimension

```python
x = torch.randn(4, 1, 3)

y = x.squeeze(1)
print(y.shape)  # torch.Size([4, 3])
```

If the dimension is not size 1, nothing happens:

```python
x = torch.randn(4, 2, 3)
y = x.squeeze(1)
print(y.shape)  # torch.Size([4, 2, 3])
```

---

## 2. `torch.unsqueeze()`

### Purpose

Adds a dimension of size **1** at a specified index.

### Basic Example

```python
x = torch.randn(4, 3)
print(x.shape)  # torch.Size([4, 3])

y = x.unsqueeze(0)
print(y.shape)  # torch.Size([1, 4, 3])
```

Adds a new dimension at index 0.

---

### Add Dimension at Other Positions

```python
x = torch.randn(4, 3)

x.unsqueeze(1).shape  # torch.Size([4, 1, 3])
x.unsqueeze(2).shape  # torch.Size([4, 3, 1])
```

The argument is the **dimension index** where the new axis is inserted.

---

## 3. Relationship to Shape and Indexing

If:

```python
x.shape == (4, 3)
```

Then:

| Operation        | Result Shape |
| ---------------- | ------------ |
| `x.unsqueeze(0)` | (1, 4, 3)    |
| `x.unsqueeze(1)` | (4, 1, 3)    |
| `x.unsqueeze(2)` | (4, 3, 1)    |

Dimension index refers to the position in the shape tuple.

---

## 4. Why This Matters (Very Important in DL)

### (a) Batch dimension

Neural networks expect input like:

```
(batch, channels, height, width)
```

If you have a single image:

```python
img.shape == (3, 256, 256)
```

You must add batch dimension:

```python
img = img.unsqueeze(0)
# (1, 3, 256, 256)
```

---

### (b) Broadcasting

Example:

```python
x = torch.randn(4, 3)
bias = torch.randn(3)

x + bias  # works because broadcasting
```

If bias shape is incompatible:

```python
bias = torch.randn(4)
```

You may need:

```python
bias = bias.unsqueeze(1)
```

---

## 5. Equivalent Alternatives

You can also use indexing syntax:

```python
x = torch.randn(4, 3)

x[None, :, :]     # same as unsqueeze(0)
x[:, None, :]     # same as unsqueeze(1)
```

---

## 6. Important: Memory Behavior

* `squeeze()` and `unsqueeze()` do **not copy data**
* They only change tensor metadata (view operation)
* Very cheap operations

---

## 7. Common Mistakes

### Mistake 1: Accidentally removing batch dimension

```python
x = torch.randn(1, 3, 256, 256)
x = x.squeeze()
```

Now shape becomes:

```
(3, 256, 256)
```

Model will break if it expects batch dimension.

Better: Alwasy specify which dim to move.

```python
x.squeeze(0)
```

---

### Mistake 2: Confusing `unsqueeze` with reshape

`unsqueeze()` only inserts size-1 dimension.
It does NOT rearrange elements.

---

## 8. Mental Model

* `unsqueeze(dim)` → insert axis at position `dim`
* `squeeze(dim)` → remove axis at `dim` if size == 1
* Pure shape manipulation, no data movement

---

If you want, I can also prepare a short section explaining how `squeeze/unsqueeze` interact with broadcasting rules step-by-step.

