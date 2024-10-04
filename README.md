# BDF-NET

BDF-NET is a deep learning framework designed for Bidirectional dynamic frame prediction network for total-body  [68Ga]Ga-PSMA-11 and [68Ga]Ga-FAPI-04 PET images. This repository contains the code for training, testing, and customizing the network architecture.

## Project Structure

```plaintext
📂 BDF-NET
├── 📁 data
│   ├── 📄 base_dataset.py
│   ├── 📄 dummy_dataset.py
│   ├── 📄 image_folder.py
│   ├── 📄 myDataloader.py
│   ├── 📄 __init__.py
├── 📁 models
│   ├── 📄 base_model.py
│   ├── 📄 mynetwork.py
│   ├── 📄 my_model.py
│   ├── 📄 networks.py
│   ├── 📄 test_model.py
│   ├── 📄 __init__.py
├── 📄 mytrain.py
├── 📁 options
│   ├── 📄 base_options.py
│   ├── 📄 test_options.py
│   ├── 📄 train_options.py
│   ├── 📄 __init__.py
├── 📄 README.md
├── 📄 test.py
├── 📁 util
│   ├── 📄 EMA.py
│   ├── 📄 get_data.py
│   ├── 📄 html.py
│   ├── 📄 image_pool.py
│   ├── 📄 util.py
│   ├── 📄 visualizer.py
│   ├── 📄 __init__.py
```
## Usage
You can specify your own configuration file by modifying the options in the 📁options folder.
### Training
To train the model, run:
```
python mytrain.py --config <config_file>
```
### Test
To test the model, run:
```
python test.py --config <config_file>
```

