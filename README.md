# Forvia_Project

## Installation guide

For Windows:

1. Download and install **Docker Desktop** (https://www.docker.com/products/docker-desktop/)
2. Download and Install **WSL** (Windows 11 recommmended for WSL2, Hyper-V should not be used with Windows 11, https://learn.microsoft.com/en-us/windows/wsl/install)
3. Install **Ubuntu** in the Linux Subsystem if it is not installed per default (**Any other distribution can be used, but is not tested!** )
4. Set up Docker Desktop
5. Go to the Docker Engine settings and check if WSL and the installed Linux Distro (Ubuntu in this case) is used by Docker when running the engine
   1. If not used, check the Distro you want to use and restart the engine
6. Continue with the Docker Manual


For Linux:

1. Follow the Steps included in the official installations guide for Ubuntu https://docs.docker.com/engine/install/ubuntu/ or any other distribution you are using https://docs.docker.com/engine/install/
2. Continue with the Docker Manual

## Docker Manual

### Step 1

Open the terminal in your IDE or use Powershell (or whichever you prefer) in the directory where the Dockerfile is situated

### Step 2

To build and run the docker image use the following commands in you terminal:

```bash
docker build --no-cache -t your-image-name:latest .
```
```bash
- docker run -it --rm your-image-name:tag
```

*Note: Replace your-image-name with whatever you like*

### Manual

- TODO