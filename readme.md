# Dibimbing Data Engineering Callenges

Repository for Data Engineering challenge from Dibimbing
## Prerequisite
- ### Requirements
    - Install Make ~ `sudo apt install make`
    - Install Docker ~ `sudo apt install docker`
    - Install Docker Compose ~ `sudo apt install docker-compose`
- ### Deployment Scripts
    - I have grouped some of the deployment scripts into a make commands available at `Makefile`, the `help` options is available upon command `sudo make help`
        ```bash
        ## postgres		- Run a Postgres container, including its inter-container network. 

        ## spark		- Run a Spark cluster, rebuild the postgres container, then create the destination tables 

        ## jupyter		- Spinup jupyter notebook for testing and validation purposes.

        ## clean		- Cleanup all running containers related to the challenge.
        ```

            Please note that this repo is using docker to build up a development environment, thus it may take a couple of minutes on pulling images from the hub. 

## Cleanup
- For cleanup the containers, run `sudo make clean`

## Add-ons
- I include my exploration notebook in the `notebooks` folder. To run jupyter, run `sudo make jupyter` and wait until the token shown.

`Owner :    Thosan Girisona Suganda`