# LetsPlay4Emotion: Domain Adaptation Methods for Emotion and Pain Recognition via Video Games

Seeing the patient’s emotional and physical condition is crucial when designing patient-computer interaction systems. However, gathering large datasets in sensitive situations like filming a person in pain can be challenging and ethically questionable. 

The primary aim of this study is to assess the possibility of using synthetic data as an alternative data source to create models capable of effectively recognizing patient pain. Initially, a synthetic dataset was generated as the foundation for model development. To maintain the relevance of the synthetically generated dataset’s diversity, a 3D model of real people was created by extracting facial landmarks from a source dataset and generating 3D meshes using [EMOCA (Emotion Driven Monocular Face Capture and Animation)](https://github.com/radekd91/emoca). Meanwhile, facial textures were sourced from publicly available datasets like [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ.git) and [FFHQ-UV](https://github.com/csbhr/FFHQ-UV). 

An efficient pipeline was created for human mesh and texture generation, resulting in a dataset of 8,600 synthetic human heads generated in approximately 2 hours per perspective and texture. The datasets encompass varying facial textures and perspectives and total over 300 GB. This approach enhances gender and ethnic diversity while introducing perspectives from previously unseen viewpoints. 

Combining the 3D models with the extracted textures created new characters with varying facial textures but identical facial expressions. The study aims to bridge the gap between synthetic data and real-world medical contexts using domain adaptation methods, like Domain Mapping. This approach eliminates the need for human participants and addresses ethical issues associated with traditional data collection methods. 

Different combinations of datasets, encompassing various textures and perspectives, were utilized to train models and assess the feasibility of synthetic data for domain adaptation (Domain Mapping) with real human data as input video. 

However, incorporating synthetic and real data leads to improved pain recognition capabilities. This combined approach can leverage the strengths of both real and synthetic datasets, resulting in a more robust and effective model for pain recognition.

To generate the meshes, we use the EMOCA repository. For creating the textures, we utilize FFHQ-UV. The video rendering is done using Blender.

##Installation 
This code employs distinct conda environments for the respective repositories, [FFHQ-UV](https://github.com/csbhr/FFHQ-UV) and [EMOCA](https://github.com/radekd91/emoca), for their specific generation tasks.

###Dependencies
1) Install [conda](https://docs.anaconda.com/free/miniconda/)
2) Run `pull_submodules.sh` for the submodules
###Mesh-Generation
Follow the instruction in the [EMOCA](https://github.com/radekd91/emoca/tree/release/EMOCA_v2/gdl_apps/EMOCA#installation) repository
###Texture-Generation
Follow the instruction in the [FFHQ-UV](https://github.com/csbhr/FFHQ-UV/tree/main?tab=readme-ov-file#dependencies) repository
###Video-Rendering
(Due the use of a Slurm Cluster management system, a container system was used for running the blender in a parallelized way) \
For rendering a sequence of meshes the Blender plugin [Stop-motion-OBJ](https://github.com/neverhood311/Stop-motion-OBJ/releases) is used
1) Download the latest version which is compatible with Blender LTS 3.6
2) For easier usage download [Blender LTS 3.6](https://www.blender.org/download/lts/3-6/) on the local device and install the plugin
3) After successful installing the plugin and Blender, save the config folder from blender for later mounting it in the container \
   (for the location of the folder see [here](https://blender.stackexchange.com/a/82))
4) create inside a folder these folders with the followings files/folders:
* `blender` : the .blend files with the current configuration of camere perspective, lightning, etc., the Stop-motion-OBJ folder and the config folder
* `all_mesh`: for the mesh files
* `render`: the render scripts are inside
* `ffhq_textures`: the texture files
* `videos_mesh`: output folder where the rendered videos are saved

Change the mounted folder(-structure) in the [`generating_threading.py`](render/generating_threading.py) between line 57 and 64 for other folder structure
###Model Training
Install the conda `environment_model.yml`

##Usage
The scripts are adjusted for the [Slurm cluster](https://slurm.schedmd.com/)

###Mesh-Generation
Use and/or adjust the [`create_emoca_mesh.sh`](mesh/create_emoca_mesh.sh) script 

###Texture-Generation
1) Create Texture: \
   run [`create_texture.sh`](texture/create_texture.sh) with the arguments `--input_dir` and `--output_dir`
2) Apply UV mapping with the texture and the meshes: \
   run [`apply_texture_with_mesh.sh`](texture/apply_texture_with_mesh.sh) and adjust the arguments `--input_dir` and `--thread_num` (this is an inplace operation)
###Video-Rendering
(For render videos a .blend file is needed where the camera, the lightning, etc. is set) \
Run the [`start_render_mesh.sh`](render/start_render_mesh.sh) and enter the parameter



