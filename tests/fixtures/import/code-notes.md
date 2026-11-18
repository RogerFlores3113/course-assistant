# Lecture 3: Loops and lists

Here is a short loop. It must survive chunking exactly.

```python
def total(xs):
    t = 0

    for x in xs:
        t += x


    return t
```

Then a second example with a blank-line-heavy body.

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)
```
