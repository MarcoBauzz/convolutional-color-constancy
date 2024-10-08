# A Convolutional Framework for Color Constancy

Official implementation of:
Buzzelli, Marco, and Simone Bianco. "A Convolutional Framework for Color Constancy." IEEE Transactions on Neural Networks and Learning Systems (2024).

```
@article{buzzelli2024convolutional,
  title={A Convolutional Framework for Color Constancy},
  author={Buzzelli, Marco and Bianco, Simone},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```

## Training

TODO

## Running replica of Low-level framework by Van der Weijer et al.

### WP
`python cfcc.py --mode='test' --epochs=0 --name-exp=expTMP --validation-transform=none --W=-1 --W-last=1 --njet=0 --intermediate-blocks=0 --mink-norm=-1 --sigma=0 --nl=none --njet-protection1=none --njet-protection2=none --mink-protection-pow=none`

### GW
`python cfcc.py --mode='test' --epochs=0 --name-exp=expTMP --validation-transform=none --W=-1 --W-last=1 --njet=0 --intermediate-blocks=0 --mink-norm=1 --sigma=0 --nl=none --njet-protection1=none --njet-protection2=none --mink-protection-pow=none`

### SoG
`python cfcc.py --mode='test' --epochs=0 --name-exp=expTMP --validation-transform=none --W=-1 --W-last=1 --njet=0 --intermediate-blocks=0 --mink-norm=5 --sigma=0 --nl=none --njet-protection1=none --njet-protection2=none --mink-protection-pow=none`

### GGW
`python cfcc.py --mode='test' --epochs=0 --name-exp=expTMP --validation-transform=none --W=-1 --W-last=1 --njet=0 --intermediate-blocks=0 --mink-norm=5 --sigma=2 --nl=none --njet-protection1=none --njet-protection2=none --mink-protection-pow=none`

### GE1
`python cfcc.py --mode='test' --epochs=0 --name-exp=expTMP --validation-transform=none --W=-1 --W-last=1 --njet=1 --intermediate-blocks=0 --mink-norm=5 --sigma=2 --nl=none --njet-protection1=none --njet-protection2=none --mink-protection-pow=none`

### GE2
`python cfcc.py --mode='test' --epochs=0 --name-exp=expTMP --validation-transform=none --W=-1 --W-last=1 --njet=2 --intermediate-blocks=0 --mink-norm=5 --sigma=2 --nl=none --njet-protection1=none --njet-protection2=none --mink-protection-pow=none`
