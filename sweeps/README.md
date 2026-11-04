# Sweep specs

The specs here train the final method; `scripts/sweep.py <spec>` runs each through `solfire`.
Everything that was tried and dropped is in `experiments/` and runs through `solfire-lab`
(see `src/solfire/experiments/README.md`).

In order:

1. `scripts/pretrain_mim.py` pretrains the ResNet34 encoder on the observations (`runs/ssl/mim-unetpp-r34`).
2. Five-fold models, which give the out-of-fold maps that the post-processing is calibrated on and the
   crop refiner is trained on:
   - `cv5_ssl.json`, `cv5_mitb2.json`, `cv5_hrnet.json`: 40 epochs of BCE + Dice
   - `finetune_ssl.json`, `finetune_mitb2.json`, `finetune_hrnet.json`: 8 epochs with the Lovász hinge added
   - `finetune_zoom_hrnet.json`: 12 epochs of the HRNet at 2× input scale
3. Full-data models, trained on every record and shipped (`weights/final/`):
   - `full.json`: MiT-B2 and HRNet-W32, 40 epochs, then the Lovász fine-tune (its ResNet34 ImageNet
     configs are not shipped; the SSL one replaced them)
   - `full_ssl.json`: the SSL ResNet34, 40 epochs, then the Lovász fine-tune
   - `full_zoom.json`: the 2× zoom HRNet, fine-tuned from the full-data HRNet
