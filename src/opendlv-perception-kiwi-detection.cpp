/*
 * Copyright (C) 2018  Christian Berger
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/dnn.hpp>

#include <cstdint>
#include <tuple>
#include <iostream>
#include <memory>
#include <mutex>
#include <string.h>

using namespace cv;
using namespace cv::dnn;

// Confidence interval and other constants
float confThreshold = 0.5f; // Confidence threshold
float nmsThreshold = 0.4f;  // Non-maximum suppression threshold
int widthInput = 416;       // Width of network's input image
int heightInput = 416;      // Height of network's input image

//Classes declaraion
std::vector<std::string> classes;

// Postprocess declaraion
std::tuple <uint16_t, uint16_t, uint16_t, uint16_t> postprocess(Mat &frame, const std::vector<Mat>& out);

//Draw predictionbox
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

//std::vector<std::string> getOutputsNames(const Net& net);

int32_t main(int32_t argc, char **argv)
{
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ((0 == commandlineArguments.count("cid")) ||
        (0 == commandlineArguments.count("name")) ||
        (0 == commandlineArguments.count("width")) ||
        (0 == commandlineArguments.count("height")))
    {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=640 --height=480 --verbose" << std::endl;
    }
    else
    {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        const bool VERBOSE{commandlineArguments.count("verbose") != 0};

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid())
        {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            // Interface to a running OpenDaVINCI session; here, you can send and receive messages.
            cluon::OD4Session od4{static_cast<uint16_t>(std::stoi(commandlineArguments["cid"]))};

            // Handler to receive distance readings (realized as C++ lambda).
            std::mutex distancesMutex;
            float front{0};
            float rear{0};
            float left{0};
            float right{0};
            auto onDistance = [&distancesMutex, &front, &rear, &left, &right](cluon::data::Envelope &&env) {
                auto senderStamp = env.senderStamp();
                // Now, we unpack the cluon::data::Envelope to get the desired DistanceReading.
                opendlv::proxy::DistanceReading dr = cluon::extractMessage<opendlv::proxy::DistanceReading>(std::move(env));

                // Store distance readings.
                std::lock_guard<std::mutex> lck(distancesMutex);
                switch (senderStamp)
                {
                case 0:
                    front = dr.distance();
                    break;
                case 2:
                    rear = dr.distance();
                    break;
                case 1:
                    left = dr.distance();
                    break;
                case 3:
                    right = dr.distance();
                    break;
                }
            };
            // Finally, we register our lambda for the message identifier for opendlv::proxy::DistanceReading.
            od4.dataTrigger(opendlv::proxy::DistanceReading::ID(), onDistance);
            
            // The names of the object to be identified: Here it is kiwi
            std::string classesFile = "/trained_data/obj.names";
            std::ifstream ifs(classesFile.c_str());
            std::string line;
            while (getline(ifs, line))
            {
                classes.push_back(line);
            }
            
            // Configuration and the weight file
            std::string modelConfiguration = "/trained_data/yolov3-tiny.cfg";
            std::string modelWeights = "/trained_data/yolov3-tiny_final.weights";
            //std::string modelWeights = "/trained_data/yolov3-tiny_sim.weights";

            // Darknet configuration
            Net net = readNetFromDarknet(modelConfiguration, modelWeights);
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();

            // Endless loop; end the program by pressing Ctrl-C.
            while (od4.isRunning())
            {
                cv::Mat img;

                // Wait for a notification of a new frame.
                sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    // Copy image into cvMat structure.
                    // Be aware of that any code between lock/unlock is blocking
                    // the camera to provide the next frame. Thus, any
                    // computationally heavy algorithms should be placed outside
                    // lock/unlock
                    cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
                    img = wrapped.clone();
                }
                sharedMemory->unlock();

                // TODO: Do something with the frame.

                // Converting the image from a 4 channel to a 3 channel 
                cvtColor(img, img, COLOR_RGBA2RGB);

                Mat blob;
                blobFromImage(img, blob, 1/255.0, Size(widthInput, heightInput), true, false);
                net.setInput(blob);

                std::vector<Mat> outs;
                net.forward(outs, outNames);

                auto [centre_x_msg, centre_y_msg, width_msg, height_msg] = postprocess(img, outs);
                
                // Converting the global co-ordinates to local
                {
                   centre_x_msg = centre_x_msg - WIDTH/2; // This makes the Y axis pass though the kiwi
                   centre_y_msg = HEIGHT - centre_y_msg; // This makes the X axis pass through the origin
                   // Width and height of the detection remain unchanged.
                }
                
                
                

                // Display image.
                if (VERBOSE)
                {

                    cv::imshow(sharedMemory->name().c_str(), img);
                    cv::waitKey(1);
                }

                ////////////////////////////////////////////////////////////////
                // Do something with the distance readings if wanted.
                {
                    std::lock_guard<std::mutex> lck(distancesMutex);
                    std::cout << "front = " << front << ", "
                              << "rear = " << rear << ", "
                              << "left = " << left << ", "
                              << "right = " << right << "." << std::endl;
                }

                ////////////////////////////////////////////////////////////////
                // Example for creating and sending a message to other microservices; can
                // be removed when not needed.
                opendlv::logic::perception::KiwiDetection kiwiDetected;
                kiwiDetected.xCenter(centre_x_msg);
                kiwiDetected.yCenter(centre_y_msg);
                kiwiDetected.width(width_msg);
                kiwiDetected.height(height_msg);
                od4.send(kiwiDetected);

                ////////////////////////////////////////////////////////////////
                // Steering and acceleration/decelration.
                //
                // Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
                // Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
                //opendlv::proxy::GroundSteeringRequest gsr;
                //gsr.groundSteering(0);
                //od4.send(gsr);

                // Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
                // Be careful!
                //opendlv::proxy::PedalPositionRequest ppr;
                //ppr.position(0);
                //od4.send(ppr);
            }
        }
        retCode = 0;
    }
    return retCode;
}

// Defining the postprocess function
std::tuple <uint16_t, uint16_t, uint16_t, uint16_t> postprocess(Mat& frame, const std::vector<Mat>& outs)
{
    
    // Remove the bounding boxes with low confidence using non-maxima suppression
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    float max_confidence = 0.0f;
    uint16_t centre_x_fun = 0;
    uint16_t centre_y_fun = 0;
    uint16_t width_fun = 0;
    uint16_t height_fun = 0;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate boxes with lower confidences
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        
        if (confidences[idx] > max_confidence && box.height<250)
        {
            drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
            max_confidence = confidences[idx];
            centre_x_fun = box.x + box.width / 2;
            centre_y_fun = box.y + box.height / 2;
            width_fun = box.width;
            height_fun = box.height;
            std::cout<<width_fun<<std::endl;
        }
    }
    //static_cast<uint16_t>(centre_y)
    return {centre_x_fun, centre_y_fun, width_fun, height_fun};
}

// Drawing the prediction box for the detection.
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    // ----------------------------------
    // Draw the predicted bounding box
    // Draw a rectangle displaying the bounding box
    // ----------------------------------
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255),3);

    //Get the label for the class name and its confidence
    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
}
