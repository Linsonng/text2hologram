# text2hologram

`text2hologram` is an advanced Python package engineered to create holograms directly from textual input. Leveraging the power of libraries like [odak](https://github.com/kaanaksit/odak), timm, transformers, diffusers, accelerate, and torch, it provides a seamless way to visualize your text in a whole new dimension.

This package is compatible with Python 3.9 and above.

For an in-depth look into the code and package structure, feel free to explore the source code here: [Source Code](https://github.com/Linsonng/text2hologram/tree/main/text2hologram/src/text2hologram)

You are also welcome to run through Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linsonng/text2hologram/blob/main/text2hologram.ipynb)

## Installation
To install the text2hologram package, use the following pip command:

```shell
pip install text2hologram
```



## Usage

Invoke `text2hologram` from the command line to use it. Here's a simple demonstration:

```shell
text2hologram --device 'cuda' --outputdir './output'
```

In the command above, `text2hologram` starts by verifying the specified device. Upon successful verification, it prompts for a text input, transforms the text into a hologram using the GPU (as specified by 'cuda'), and saves the resultant hologram in the './output' directory.

The options provided in the command are as follows:

- `--device` : The computing device to be used ('cpu' or 'cuda').
- `--outputdir` : The output directory to store the generated holograms.
- `--inference_steps` : The number of inference steps to be used for generating the image. Higher the number, more detailed and accurate the hologram. However, it would proportionately increase the computation time and cost.

For additional settings and options, refer to the [config.json](https://github.com/Linsonng/text2hologram/blob/main/text2hologram/src/text2hologram/config.json) file. All settings are named in a self-explanatory manner for ease of use.

## Contact

Should you have any queries or require further clarification, don't hesitate to get in touch with [Pengze Li](https://linsonng.github.io/)  at linsonng@163.com.
