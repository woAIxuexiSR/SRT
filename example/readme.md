# SRT Scene Configuration File

Welcome to the readme for the scene configuration file of SRT! This document serves as your comprehensive guide to understanding the structure and available options within the scene configuration file. By following this guide, you will gain the necessary knowledge to effectively build and utilize SRT for your ray tracing needs.

## 1. File Structure

The SRT scene configuration file is structured as a JSON document. It allows you to specify all the necessary rendering parameters.

The file consists of three sections: "render," "passes," and "scene." The "render" section serves to define the core parameters of the renderer. The "passes" section is dedicated to defining the render passes to be employed. Lastly, the "scene" section encompasses all the scene properties.

## 2. Render Section

- **type** : The type of the render. Valid values are "image", "interactive", or "video".

- **output** : The output file name. Note that this option will be ignored if the render type is set to "interactive".

- **resolution** : The resolution of the output image. It should be specified as `[width, height]`. However, if the scene file is in pbrt format, the resolution value will be disregarded.

- **frame** : The frame number of the video. This option will be ignored if the render type is set to "image".

## 3. Passes Section

The passes section is an array of passes, where each pass is defined as an object with the following properties:

- **name** : The name of the pass.

- **enable** : A boolean value indicating whether the pass should be enabled.

- **params** : An object containing the parameters of the pass. Depending on the pass type. For more details, please refer to the [passes source code](../src/passes/).

## 4. Scene Section

- **model** : An object that encompasses the model properties.

  - **type** : The type of the model. Valid values are "pbrt" (only support a subset) or any model file extensions supported by Assimp.

  - **path** : The path to the model file.

- **camera** : An object that contains the camera properties. Note that this section may be ignored if the file format is set to pbrt. The camera object includes the following properties:

  - **type** : The type of the camera. Valid values are "perspective," "orthographic, "thinlens", or "environment".

  - **position** : The position of the camera as a three-dimensional vector `[x, y, z]`.

  - **target** : The target of the camera as a three-dimensional vector `[x, y, z]`.

  - **up** : The up vector of the camera as a three-dimensional vector `[x, y, z]`.

  - Other properties: Contains additional properties specific to different camera types. For more detailed information, please refer to the [cameras source code](../src/device_include/scene/camera/).

- **environment** : An object that encompasses the environment properties. Note that this section may be ignored if the file format is set to pbrt. The environment object includes the following properties:

  - **type** : The type of the environment. Valid values are "constant," "uvmap," or "cubemap".

  - **color** : For "constant" environment type, specifies the color of the environment as a three-dimensional vector `[r, g, b]`.

  - **path** : For "uvmap" environment type, specifies the path to the uvmap file. For "cubemap" environment type, specifies an array of paths to the cubemap files.

## 5. Example Scene Configuration File

```json
{
    "render": {
        "type": "image",
        "output": "output.png"
    },
    "passes": [
        {
            "name": "PathTracer",
            "enable": true,
            "params": {
                "max_depth": 5,
                "sample_per_pixel": 128,
                "use_nee": true
            }
        }
    ],
    "scene": {
        "model": {
            "type": "pbrt",
            "path": "cornell_box.pbrt"
        }
    }
}
```