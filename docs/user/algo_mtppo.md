# Multi-Task Proximal Policy Optimization (Multi-Task PPO)

```eval_rst
.. list-table::
   :header-rows: 0
   :stub-columns: 1
   :widths: auto

   * - **Paper**
     - Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning :cite:`yu2019metaworld`, Proximal Policy Optimization Algorithms :cite:`schulman2017proximal`
   * - **Framework(s)**
     - .. figure:: ./images/pytorch.png
        :scale: 10%

        PyTorch
   * - **API Reference**
     - `garage.torch.algos.PPO <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.PPO>`_
   * - **Code**
     - `garage/torch/algos/ppo.py <https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/ppo.py>`_
   * - **Examples**
     - :ref:`mtppo_metaworld_ml1_push`, :ref:`mtppo_metaworld_mt10`, :ref:`mtppo_metaworld_mt50`
```

Multi-Task PPO is a multi-task RL method that aims to learn PPO algorithm to maximize the average expected return acorss multiple tasks. The algorithm is evaluated on their average performance over training tasks,instead of seperate test sets of tasks.

## Examples

### mtppo_metaworld_ml1_push

```eval_rst
.. literalinclude:: ../../examples/torch/mtppo_metaworld_ml1_push.py
```

### mtppo_metaworld_mt10

```eval_rst
.. literalinclude:: ../../examples/torch/mtppo_metaworld_mt10.py
```

### mtppo_metaworld_mt50

```eval_rst
.. literalinclude:: ../../examples/torch/mtppo_metaworld_mt50.py
```

## References

```eval_rst
.. bibliography:: references.bib
   :style: unsrt
   :filter: docname in docnames
```

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*
