# Contributing

Yes, please. The first outside PR here fixed a bug where steering silently did nothing. Standards are therefore high and gratitude is real.

```
pytest                    # unit tests, CPU is fine
pytest -m integration     # the real engine, needs a GPU
python -m hotwire.verify  # before any PR that touches the patch: steering changes the output, unsteered requests don't
```

- Say which vLLM version you ran. Half of this plugin's life is vLLM changing underneath it.
- "It does nothing and says nothing" is the worst kind of bug here. If you find one, you get the changelog line.
- Want something to do? [the open questions](https://github.com/moudrkat/moudrkat/blob/main/OPEN-QUESTIONS.md): relative strength and a read operation are both waiting.

I answer within a day.
