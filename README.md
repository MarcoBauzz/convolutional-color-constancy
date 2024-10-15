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

### LCF-A
`python cfcc.py --name-exp=exp101L1_LCFbase_r1_t1_e1k --image-size=200 --intermediate-blocks=0 --pool-size=1 --W-last=1 --W=3 --sigma=-1 --test-sets=1 --save-output --njet=1 --inner-size=9 --mink-norm=1.000001 --epochs=1000 --njet-protection1=prelu --nl=prelu --njet-protection2=posrelu --mink-protection-pow=oneabs --validation-transform=isotropic --data-augmentation`

### ECF-A
`python cfcc.py --name-exp=exp101L1_ECFbase_r1_t1_e1k --image-size=200 --intermediate-blocks=1 --pool-size=1 --W-last=1 --W=3 --sigma=-1 --test-sets=1 --save-output --njet=1 --inner-size=9 --mink-norm=1.000001 --epochs=1000 --njet-protection1=prelu --nl=prelu --njet-protection2=posrelu --mink-protection-pow=oneabs --validation-transform=isotropic --data-augmentation`

### ECF-B
`python cfcc.py --name-exp=exp101L1_ECFswept_r1_t1_e1k --image-size=200 --batch-size=8 --intermediate-blocks=1 --pool-size=3 --W-last=2 --W=5 --sigma=-1 --test-sets=1 --save-output --njet=1 --inner-size=27 --mink-norm=1.000001 --epochs=1000 --njet-protection1=prelu --nl=prelu --njet-protection2=posrelu --mink-protection-pow=oneabs --validation-transform=isotropic --data-augmentation`


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
