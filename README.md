# opendlv-perception-kiwi-detection-cpp
An OpenDLV microservices program developed for an autonomous robot to detect other robot in its pathway. 

OpenDLV is a modern open source software developed by [Chalmers Revere](https://github.com/chalmers-revere/opendlv) for self-driving vehicles. 

[Kiwi](https://github.com/chalmers-revere/opendlv-tutorial-kiwi) is miniature robotic vehicle platform from Chalmers Revere.

This is a part of the TME290 - Autonomous Robots course.

The final output can be visualised in the following video:\
[![Alt text](https://img.youtube.com/vi/fW5CsNZUt0I/0.jpg)](https://www.youtube.com/watch?v=fW5CsNZUt0I)

# How to get the program running:

Pre-requisite:
1. Docker must be installed.

################################### Part A: ##############################################

This is to remove any images and containers that can create a conflict with the program.

Caution: This will wipe out all the containers so do it only if it is necessary.

1. Stop all running containers using:\
docker stop $(docker ps -a -q)

2. Clear all containers:\
docker rm $(docker ps -a -q)

3. Remove all images:\
docker rmi -f $(docker images -a -q)


################################### Part B: ##############################################

This is to get all the files needed to the local system.

1. Pull the code to your PC/Laptop using
   git clone https://github.com/Rahul-Pi/opendlv-perception-kiwi-detection-cpp.git

################################### Part C: ##############################################

This is to build the docker from the files.

1. Open a terminal in the folder "opendlv-perception-kiwi-detection-cpp" and run the following command:\
   docker build -f Dockerfile -t opendlv-perception-kiwi-detection-cpp .


################################### Part D: ##############################################

This part is to run the file and record the output.

1. Open a terminal in the folder "opendlv-perception-kiwi-detection-cpp" and run the following code to start the replay viewer.\
   docker-compose -f h264-replay-viewer.yml up
2. Launch another terminal in the same folder "opendlv-perception-kiwi-detection-cpp" and run the following code to start the kiwi detection.\
   docker-compose -f simulate-kiwi-detection.yml up

