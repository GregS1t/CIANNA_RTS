# CIANNA REAL TIME SERVICE

This document outlines the steps necessary to install and configure all modules and dependencies for the CIANNA_RTS project on client and server. Some parts are redundant to make both section independant.
Furure version of the repo will separate Client and Server doc... For the moment, it's good enough.

## 1. Installation on the Client side

### Install Git (if not already installed)
Git is necessary to pull the codes on your computer.

#### For Ubuntu/Debian
```shell
$sudo apt updade && sudo apt install git
```
#### Mac OS
To be completed ... 

#### Windows
To be completed ...


### Install python 

Download: [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)

### Clone the code

Clone the project repository: [https://gitlab.obspm.fr/gsainton/cianna_RTS](https://gitlab.obspm.fr/gsainton/cianna_RTS)

```shell
$git clone https://gitlab.obspm.fr/gsainton/cianna_RTS
```

### Adapt the different path 

1. Modify the `param_cianna_rts_client.json` file if necessary. The file is in `C_client` directory

```JSON
{
    "SERVER_URL":"http://127.0.0.1:3000",
    "URL_MODELS":"http://127.0.0.1:3000/models/CIANNA_models.xml",
    "LOCAL_FILE_MODELS": "models/CIANNA_models.xml"
}
```


___

## 2. Installation on the Server side
The installation was made on Ubuntu 24.04 LTS.

### Install Git (if not already installed)

```shell
$sudo apt updade && sudo apt install git
```
### Install python 

Download: [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)

### Clone the code 
The server is implemented in the file server_cianna_rts.js.

Clone the project repository: [https://gitlab.obspm.fr/gsainton/cianna_RTS](https://gitlab.obspm.fr/gsainton/cianna_RTS)

```shell
$git clone https://gitlab.obspm.fr/gsainton/cianna_RTS
```

### Install Node.js
```shell
$sudo apt updade && sudo apt install nodejs
```
One can check the version of the Node *via* the command 

```shell
$node -v
```
Then install the Node packages manager *npm*

```shell
$sudo apt install npm
```

### Install CIANNA

The github repository of CIANNA is here: [https://github.com/Deyht/CIANNA](https://github.com/Deyht/CIANNA)

The full install procedure is here in the wiki: [https://github.com/Deyht/CIANNA/wiki](https://github.com/Deyht/CIANNA/wiki)

### Custom paths to your local needs

In `C_SERVER/yolo_cianna_detect/params`, open the file `yolo_rts.json`, one must modify the following paths : 

```JSON
    "PATH2CIANNA": "/path/2/CIANNA/src/build/lib.linux-x86_64-cpython-311",
    "CIANNA_RTS_DIR": "/path/2/CIANNA_RTS/C_SERVER",
```
---

## Start the server session

For the moment, we are in prototyping mode. The session is started by hand : 

Go in `C_SERVER/` directory with your terminal then write the following command:

```shell
$node server_cianna_rts.js
```

The programm should start like :

```shell
Server listening:  http://127.0.0.1:3000
```


- The server is waiting for a FITS file and model name+parameters
- Once everithing is valid, YOLO-CIANNA starts detecting sources
- Server returns the list of sources detected.

---
## From the client side
At the end, everything should be embedded into a GUI but for now, the execution is made by hand, so go into `C_client` directory

```shell
$python test_process_xml.py <path/to/fits_image>
```
Then, the code will go though the following steps:
- Check on the server if a new version of `CIANNA_models.xml` is available. It there is one, it will be downloaded in `models` directory ;
- This will be one of the model used for the inference on your image ;
- Check if the image is a FITS file ;
- Then both FITS file and model name are sent to the server ;
- If FITS file is valid, the YOLO-CIANNA starts to detect the sources ;
- The file of sources is sent back to the client.

This prototype is multi-threaded of the tests. Then, then detection occurs one file another the previous one. 

The next step will be to apply some priority criteria.
