# text2hologram

`text2hologram` is a Python package designed to generate holograms from text. It's built on top of the odak, timm, transformers, diffusers, accelerate, and torch libraries.

text2hologram supports Python 3.9 and above.

The project's source code is available [here](https://github.com/Linsonng/text2hologram/tree/main/text2hologram/src/text2hologram).

## Installation

To install the package, use the following pip command:

```shell
pip install text2hologram
```

## Usage

You can use the package from the command line with the `text2hologram` command. 

Here is a simple usage example:

```shell
text2hologram --device 'cuda' --outputdir './output'
```

The command above checks the device first,and then ask for a text input. generates a hologram from the input text using GPU and saves the output in the './output' directory.

Here's a brief description of each option:


- `--device` : Computation device to use. Can be 'cpu' or 'cuda'.
- `--outputdir` : Directory to save output files.
- `--inference_steps` : Inference steps to generate an image. A higher number of inference steps will result in a more accurate and detailed image, but increase the computational cost and time required.

See more settings in [config.json](https://github.com/Linsonng/text2hologram/blob/main/text2hologram/src/text2hologram/config.json). All settings are named in a self-explanatory manner.

## License

text2hologram is licensed under the MIT license. The terms of the license can be found in the LICENSE file.

## Contact

For any questions or feedback, you are welcome to reach out to Pengze Li at linsonng@163.com.
