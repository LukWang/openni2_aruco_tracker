///////////////////////////////////////////////////////////////////////////////
//
// Simple program that reads depth and color (RGB) images from Primensense
// camera using OpenNI2 and displays them using OpenCV.
//
// Ashwin Nanjappa
///////////////////////////////////////////////////////////////////////////////

#ifndef TARGET_PLATFORM_LINUX
#define TARGET_PLATFORM_LINUX 1 
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/aruco.hpp>

#include <OpenNI.h>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//#include <sensor_msgs/Image.h>


#include <iostream>
#include <sstream>
#include <fstream>

#include <math.h>

using namespace std;
using namespace cv;


class Grabber
{
public:
    void InitOpenNI();
    void InitDevice();
    void InitCalibration();
    void InitDepthStream();
    void InitColorStream();
    void displayFrames();
    void Run();
	void Exit();

private:

    void CapturePsenseDepthFrame();
    void CapturePsenseColorFrame();
    void visualize_depth(Mat&);
    void sendMarkerTf(vector<Point3f>&, vector<int>&);
    void sendMarkerTf(vector<Vec3d>& marker_trans, vector<Vec3d>& marker_rot, vector<int>& ids);

/***********marker locating functions****************/

    void locateMarker(cv::Mat &srcImage);
    void getWorldCoordinate(Point2f& image_cord, Point3f& cord);
    void getImageCoordinate(Point3f&, Point&);
    void getAllMarkerCoordinate(vector<vector<Point2f>>& corners, vector<int>& ids, vector<Point2f>& marker_center, vector<Point3f>& world_cord);
    void calculateRobotPose(vector<Point>&, vector<Point3f>&);
    void eraseRobotFromImage(Mat&, vector<Point>&, bool);
    void drawRobotJoints(Mat&, vector<Point>&);

/***************video stream variables**************/
    openni::Device*        device_;
    openni::VideoStream*   depth_stream_;
    openni::VideoStream*   color_stream_;
    openni::VideoFrameRef* depth_frame_;
    openni::VideoFrameRef* color_frame_;

    Mat color_mat;
    Mat depth_mat;
    
    Size sizeColor;
    Mat map_color1, map_color2;
/*****************marker variables*****************/
    vector<cv::Vec3d> static_marker_rvec;
    vector<cv::Vec3d> static_marker_tvec;
/*****************camera parameters*****************/
    Mat camera_matrix;
    Mat distortion_vector;
    const float fx, fy, cx, cy;


/*****************operational flags***************/
    bool printScr_color, printScr_depth;

/****************other variables******************/
    int frame_count;
    string frame_save_path;

/*******************ros variables**********************/
    ros::NodeHandle node_handle;

    image_transport::ImageTransport it;
    image_transport::Publisher depth_image_publisher;
    
    tf::TransformListener robot_pose_listener;
    tf::StampedTransform base_cam_transform;
    bool isCamBaseTransformAvailable;

    vector<std::string> joint_names;

    tf::Transform robot_pose_tansform;
    void loadRobotPoseFile(string);
    void linkToRobotTf();

    bool calibrationMode;


public:
    Grabber(bool isCalibrationMode, const ros::NodeHandle &node_handle = ros::NodeHandle()) : sizeColor(512, 424), fx(365.682), fy(365.682), cx(256.223), cy(203.687), it(node_handle)
    {
        camera_matrix = Mat::eye(3, 3, CV_32F);
        camera_matrix.at<float>(0, 0) = fx;
        camera_matrix.at<float>(1, 1) = fy;
        camera_matrix.at<float>(0, 2) = cx;
        camera_matrix.at<float>(1, 2) = cy;

        distortion_vector = Mat::zeros(1, 5, CV_64F);
        distortion_vector.at<double>(0, 0) = 0.0915081;
        distortion_vector.at<double>(0, 1) = -0.271121;
        distortion_vector.at<double>(0, 2) = 0.0;
        distortion_vector.at<double>(0, 3) = 0.0;
        distortion_vector.at<double>(0, 4) = 0.0943824;
    
        printScr_color = false;
        printScr_depth = false;

        frame_save_path = "/home/luk/catkin_ws/src/openni2_aruco_tracker/savedFrames/color_frames";
        std::vector<String> saved_frames;
    
        glob(frame_save_path + "*.jpg", saved_frames, false);
        frame_count = saved_frames.size();
        cout << "Exisiting color frame: " <<  frame_count << endl;

        /*********ros initialize*********/
        depth_image_publisher = it.advertise("openni_msgs/depth_image", 1);
    
        joint_names.push_back("shoulder_link");
        joint_names.push_back("upper_arm_link");
        joint_names.push_back("forearm_link");
        joint_names.push_back("wrist_1_link");
        joint_names.push_back("wrist_2_link");
        joint_names.push_back("wrist_3_link");

        isCamBaseTransformAvailable = false;

        calibrationMode = isCalibrationMode;
    }

};



void Grabber::InitOpenNI()
{
    auto rc = openni::OpenNI::initialize();
    if (rc != openni::STATUS_OK)
    {
        printf("Initialize failed\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }
}


void Grabber::InitDevice()
{
    device_ = new openni::Device();
    auto rc = device_->open(openni::ANY_DEVICE);
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }
}

void Grabber::InitDepthStream()
{
    depth_stream_ = new openni::VideoStream();

    // Create depth stream from device
    if (device_->getSensorInfo(openni::SENSOR_DEPTH) != nullptr)
    {
        auto rc = depth_stream_->create(*device_, openni::SENSOR_DEPTH);
        if (rc != openni::STATUS_OK)
        {
            printf("Couldn't create depth stream\n%s\n", openni::OpenNI::getExtendedError());
            exit(0);
        }
    }

    // Get info about depth sensor
    const openni::SensorInfo& sensor_info       = *device_->getSensorInfo(openni::SENSOR_DEPTH);
    const openni::Array<openni::VideoMode>& arr = sensor_info.getSupportedVideoModes();

    // Look for VGA mode in depth sensor and set it for depth stream
    for (int i = 0; i < arr.getSize(); ++i)
    {
        const openni::VideoMode& vmode = arr[i];
        if (vmode.getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_1_MM &&
            vmode.getResolutionX() == 512 &&
            vmode.getResolutionY() == 424)
        {
            depth_stream_->setVideoMode(vmode);
            break;
        }
    }

    // Start the depth stream
    auto rc = depth_stream_->start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    depth_frame_ = new openni::VideoFrameRef();
}

void Grabber::InitCalibration()
{
    const int mapType = CV_16SC2;

    initUndistortRectifyMap(camera_matrix, distortion_vector, cv::Mat(), camera_matrix, sizeColor, mapType, map_color1, map_color2);
    string pose_solution_path = "/home/luk/catkin_ws/src/openni2_aruco_tracker/robot_pose/solution_new5";
    loadRobotPoseFile(pose_solution_path);
}


void Grabber::InitColorStream()
{
    color_stream_ = new openni::VideoStream();

    if (device_->getSensorInfo(openni::SENSOR_COLOR) != nullptr)
    {
        auto rc = color_stream_->create(*device_, openni::SENSOR_COLOR);
        if (rc != openni::STATUS_OK)
        {
            printf("Couldn't create color stream\n%s\n", openni::OpenNI::getExtendedError());
            exit(0);
        }
    }

    // Get info about color sensor
    const openni::SensorInfo& sensor_info       = *device_->getSensorInfo(openni::SENSOR_COLOR);
    const openni::Array<openni::VideoMode>& arr = sensor_info.getSupportedVideoModes();

    // Look for VGA mode and set it for color stream
    for (int i = 0; i < arr.getSize(); ++i)
    {
        const openni::VideoMode& vmode = arr[i];
        if (
            vmode.getResolutionX() == 512 &&
            vmode.getResolutionY() == 424)
        {
            color_stream_->setVideoMode(vmode);
            printf("%i: %ix%i, %i fps, %i format\n", i, vmode.getResolutionX(), vmode.getResolutionY(),vmode.getFps(), vmode.getPixelFormat()); 
            //break;
        }
    }

    
    
    // Note: Doing image registration earlier than this seems to fail
    if (device_->isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
    {
        
        openni::ImageRegistrationMode mode = device_->getImageRegistrationMode();
        cout << mode << endl;
        auto rc = device_->setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
        
        if (rc == openni::STATUS_OK)
            std::cout << "Depth to color image registration set success\n";
        else
            std::cout << "Depth to color image registration set failed\n";
    }
    else
    {
        std::cout << "Depth to color image registration is not supported!!!\n";
    }

    // Start color stream
    auto rc = color_stream_->start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", openni::OpenNI::getExtendedError());
        exit(0);
    }

    color_frame_ = new openni::VideoFrameRef();
}


void Grabber::CapturePsenseDepthFrame()
{
    auto rc = depth_stream_->readFrame(depth_frame_);
    if (rc != openni::STATUS_OK)
    {
        printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
    }

    if (depth_frame_->getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM && depth_frame_->getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM)
    {
        printf("Unexpected frame format\n");
    }

    // Get pointer to Primesense depth frame
    openni::DepthPixel* dev_buf_ptr = (openni::DepthPixel*) depth_frame_->getData();

    // Copy frame data to OpenCV mat
    depth_mat = Mat(depth_frame_->getHeight(), depth_frame_->getWidth(), CV_16U, dev_buf_ptr);

    flip(depth_mat, depth_mat, 1);
    remap(depth_mat, depth_mat, map_color1, map_color2, cv::INTER_AREA);
}

void Grabber::CapturePsenseColorFrame()
{
    // Read from stream to frame
    auto rc = color_stream_->readFrame(color_frame_);
    if (rc != openni::STATUS_OK)
    {
        printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
    }

    // Pointer to Primesense color frame
    openni::RGB888Pixel* dev_buf_ptr = (openni::RGB888Pixel*) color_frame_->getData();

    // Make mat from camera data
    //cout << color_frame_->getWidth() << 'x' << color_frame_->getHeight() << endl;
    color_mat = Mat(color_frame_->getHeight(), color_frame_->getWidth(), CV_8UC3, dev_buf_ptr);

    // Convert to BGR format for OpenCV
    cv::cvtColor(color_mat, color_mat, CV_RGB2BGR);
    flip(color_mat, color_mat, 1);
    remap(color_mat, color_mat, map_color1, map_color2, cv::INTER_AREA);
}

void Grabber::locateMarker(cv::Mat &srcImage)
{
    bool publishStaticMarker = false;
    
    vector<vector<Point2f>> corners, rejected_candidates, corners_for_pose_estimation;
    vector<int> ids, ids_for_pose_estimation;
    vector<cv::Vec3d> rvecs, tvecs;
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);
    Ptr<aruco::DetectorParameters> detect_parms_ptr = aruco::DetectorParameters::create();
    //detect_parms_ptr -> doCornerRefinement = true; //in early version of openCV
    detect_parms_ptr -> cornerRefinementMethod = aruco::CORNER_REFINE_CONTOUR;
    //detect_parms_ptr -> cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;
    //detect_parms_ptr -> cornerRefinementMethod = aruco::CORNER_REFINE_NONE;
    
    detect_parms_ptr -> minCornerDistanceRate = 0.1;
    
    aruco::detectMarkers(srcImage, dictionary, corners, ids, detect_parms_ptr, rejected_candidates);
    
    aruco::drawDetectedMarkers(srcImage, corners, ids);


    for(int i = 0;i < ids.size(); i++)
    {
        if(ids[i] == 0)
        {
            corners_for_pose_estimation.push_back(corners[i]);
            corners.erase(corners.begin()+i);
            ids_for_pose_estimation.push_back(ids[i]);
        }
    }

    vector<int>::iterator it_ids;

    for(it_ids = ids.begin(); it_ids != ids.end();)
    {
        if(*it_ids == 0)
        {
            ids.erase(it_ids);
        }
        else 
            ++it_ids;
    }
    aruco::estimatePoseSingleMarkers(corners_for_pose_estimation, 0.133f, camera_matrix, distortion_vector, rvecs, tvecs);
    for(int i = 0;i < ids_for_pose_estimation.size(); ++i)
    {
        cv::aruco::drawAxis(srcImage, camera_matrix, distortion_vector, rvecs[i], tvecs[i], 0.1);
        if (ids_for_pose_estimation[i] == 0)
        {
            if (static_marker_rvec.size() >= 2)
            {
                for(int j = 0; j < static_marker_rvec.size(); j++){
                    rvecs[i] += static_marker_rvec[j];
                    tvecs[i] +=static_marker_tvec[j];
                }
                rvecs[i] /= ((double)static_marker_rvec.size() + 1.0);
                tvecs[i] /= ((double)static_marker_tvec.size() + 1.0);
                static_marker_tvec.clear();
                static_marker_rvec.clear();
                publishStaticMarker = true;
            }
            
            else{ 
                static_marker_tvec.push_back(tvecs[i]);
                static_marker_rvec.push_back(rvecs[i]);
           }
        }
    }

    if(publishStaticMarker)
        sendMarkerTf(tvecs, rvecs, ids_for_pose_estimation);
    
    tvecs.clear();
    rvecs.clear();
    corners_for_pose_estimation.clear();
    ids_for_pose_estimation.clear();


    for(int i = 0;i < ids.size(); i++)
    {
        if(ids[i] == 10)
        {
            corners_for_pose_estimation.push_back(corners[i]);
            corners.erase(corners.begin()+i);
            ids_for_pose_estimation.push_back(ids[i]);
        }
    }


    for(it_ids = ids.begin(); it_ids != ids.end();)
    {
        if(*it_ids == 10)
        {
            ids.erase(it_ids);
        }
        else 
            ++it_ids;
    }
    aruco::estimatePoseSingleMarkers(corners_for_pose_estimation, 0.133f, camera_matrix, distortion_vector, rvecs, tvecs);
    for(int i = 0;i < ids_for_pose_estimation.size(); ++i)
    {
        cv::aruco::drawAxis(srcImage, camera_matrix, distortion_vector, rvecs[i], tvecs[i], 0.1);
    }
    sendMarkerTf(tvecs, rvecs, ids_for_pose_estimation);

    //aruco::drawDetectedMarkers(srcImage, rejected_candidates); //uncomment this if want to show rejected candidates
    vector<Point2f> marker_center;
    vector<Point3f> world_cord;
    getAllMarkerCoordinate(corners, ids, marker_center, world_cord);
    sendMarkerTf(world_cord, ids);
    
    // display information on screen
    int id = 5;
    int index;
    for (index = 0; index < ids.size(); index++)
        if (id == ids[index])
        {
            ostringstream cord_text;
            Point offset (10, 10);
            cord_text << '(' << world_cord[index].x << ',' << world_cord[index].y << ',' << world_cord[index].z << ')';
            Point displayCord(marker_center[index].x, marker_center[index].y);
            circle(srcImage, displayCord, 2, Scalar(0, 0, 255), -1, 8);
            putText(srcImage, cord_text.str(), displayCord + offset, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1.5);
            cord_text.str("");
            cord_text << "id5 depth:" << depth_mat.at<uint16_t>(displayCord) << " at" << '(' << displayCord.x << ',' << displayCord.y << ')';
            putText(srcImage, cord_text.str(), Point(20,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255, 0));
            break;
        }
}

void Grabber::Run()
{
    openni::VideoStream* streams[] = {depth_stream_, color_stream_};

   while (node_handle.ok())
   {
       int readyStream = -1;
       auto rc = openni::OpenNI::waitForAnyStream(streams, 2, &readyStream, 2000);
       if (rc != openni::STATUS_OK)
       {
           printf("Wait failed! (timeout is %d ms)\n%s\n", 2000, openni::OpenNI::getExtendedError());
           break;
       }

       switch (readyStream)
       {
       case 0:
        CapturePsenseDepthFrame();
           break;
       case 1:
        CapturePsenseColorFrame();
           break;
       default:
           printf("Unxpected stream\n");
       }
        
       displayFrames();


       char c = cv::waitKey(10);
       if (' ' == c)
       {
           printScr_color = true;
           printScr_depth = true;
       }
       if ('q' == c)
           break;

   }
}

void Grabber::calculateRobotPose(vector<Point>& joint_image_cords, vector<Point3f>& joint_3d_cords)
{

    //tf::TransformListener robot_pose_listener;
    string robot_reference_frame;
    if (calibrationMode)
    {
        robot_reference_frame = "camera_base_rect";
    }
    else 
    { 
        robot_reference_frame = "camera_base";
    }


    tf::StampedTransform joint_transforms;
    tf::StampedTransform cam_base_transform;
    try
    {
        robot_pose_listener.lookupTransform(robot_reference_frame.c_str(), "base_link", ros::Time(0), cam_base_transform);
    }

    catch(tf::TransformException ex)
    {
        //ROS_ERROR("%s", ex.what());
        isCamBaseTransformAvailable = false;
        return;
    }

    isCamBaseTransformAvailable = true;
    Point3f base_location(cam_base_transform.getOrigin().x(), cam_base_transform.getOrigin().y(), cam_base_transform.getOrigin().z());
    Point base_image_cord;
    getImageCoordinate(base_location, base_image_cord);
    circle(color_mat, base_image_cord, 2, Scalar(0, 255, 255), -1, 8);
    
    ostringstream cord_text;
    cord_text.str("");
    cord_text << "base_position:" << " at" << '(' << base_location.x << ',' << base_location.y << ',' << base_location.z << ')';
    putText(color_mat, cord_text.str(), Point(20,400), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255, 0));
    
    //vector<Point3f> joint_3d_cords;
    joint_3d_cords.push_back(base_location);
    //vector<Point> joint_image_cords;
    joint_image_cords.push_back(base_image_cord);
    for(int i = 0; i < joint_names.size(); i++)
    {
        try
        {
            robot_pose_listener.lookupTransform(robot_reference_frame.c_str(), joint_names[i], ros::Time(0), joint_transforms);
        }
        catch(tf::TransformException ex)
        {
            ROS_ERROR("%s", ex.what());
            continue;
        }
        Point3f location(joint_transforms.getOrigin().x(), joint_transforms.getOrigin().y(), joint_transforms.getOrigin().z());
        joint_3d_cords.push_back(location);
        Point joint_image_cord;
        getImageCoordinate(location, joint_image_cord);
        joint_image_cords.push_back(joint_image_cord);

    }

}

void Grabber::drawRobotJoints(Mat& image, vector<Point>& joint_image_cords)
{

    
    circle(image, joint_image_cords[0], 3, Scalar(0, 255, 0), -1, 8);
    //draw joints on image
    for(int i = 1; i < (joint_image_cords.size()); i++)
    {
        circle(image, joint_image_cords[i], 3, Scalar(0, 255, 0), -1, 8);
        line(image,joint_image_cords[i-1],joint_image_cords[i], Scalar(0, 255, 255), 2);
    }
}
void Grabber::eraseRobotFromImage(Mat& erased, vector<Point>& joint_image_cords, bool isColorImage = true)
{
    float step = 1.0;
    
    for(int i = 1; i < joint_image_cords.size(); i++)
    {
        Point diff = joint_image_cords[i] - joint_image_cords[i-1];
        float slope = (float)diff.y / (float)diff.x;
        Point2f pixel_iterator((float)joint_image_cords[i-1].x, (float)joint_image_cords[i].y);
        while(norm(pixel_iterator - (Point2f)joint_image_cords[i]) > 4.0)
        {
            
            pixel_iterator.x = pixel_iterator.x + step;
            pixel_iterator.y = pixel_iterator.y + step * slope;
            Point pixel((int)pixel_iterator.x, (int)pixel_iterator.y);
            Point3f pixel3d(0.0f, 0.0f, 0.0f);
            getWorldCoordinate(pixel_iterator, pixel3d);
            
            int scale = 20;
            for (int r = pixel.x - scale; r < pixel.x + scale; r++)
                for (int c = pixel.y - scale; r < pixel.y + scale; r++)
                {
                    int diff = abs(depth_mat.at<uint16_t>(r, c) - depth_mat.at<uint16_t>(pixel));
                    if(diff < 100)
                    {
                        if (isColorImage)
                            erased.at<uint16_t>(r, c) = 0;
                        else
                            erased.at<Vec3b>(r, c) = Vec3b(255,255,255);
                    }
                }

        }
        
    }
}

void Grabber::displayFrames()
{
    vector<Point> joint_image_cords;
    vector<Point3f> joint_3d_cords;
    
    namedWindow("Color Frame");
    if(!color_mat.empty() && !depth_mat.empty())
    {
        if (printScr_color)
        {
            printScr_color = false;
            frame_count ++;
            ostringstream convert;
            convert << frame_save_path << "Frame" << frame_count << ".jpg";
            imwrite(convert.str(), color_mat);
            cout << "frame saved" << endl;
        }
        locateMarker(color_mat);

        if(!calibrationMode)
        {
            linkToRobotTf();
            
        }

        //cout << "marker located" << endl;
        
        calculateRobotPose(joint_image_cords, joint_3d_cords);
        if (!joint_image_cords.empty())
        {
            //eraseRobotFromImage(color_mat, joint_image_cords);
            drawRobotJoints(color_mat,joint_image_cords);
        }
        imshow("Color Frame", color_mat);
        
    }


//    namedWindow("Depth Frame");
    if(!depth_mat.empty())
    {
        sensor_msgs::ImagePtr depth_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", depth_mat).toImageMsg();
        depth_image_publisher.publish(depth_msg);

        if (!joint_image_cords.empty())
        {
            //eraseRobotFromImage(depth_mat, joint_image_cords, false);
        }

        Mat depth_viz;
        visualize_depth(depth_viz);
       // drawRobotJoints(depth_viz, joint_image_cords);
        
        imshow("Depth Frame", depth_viz);
    }
}

void Grabber::visualize_depth(Mat& depth_viz)
{
   if(!depth_mat.empty()) 
   {
       depth_viz = Mat(depth_mat.rows, depth_mat.cols, CV_8UC3);
       for (int r = 0; r < depth_viz.rows; ++r)
        for (int c = 0; c < depth_viz.cols; ++c)
        {
            uint16_t depth = depth_mat.at<uint16_t>(r, c);
            uint16_t level;
            uint8_t alpha;

            //sort depth information into different depth levels
            if (depth == 0)
                level = 0;
            else
                level = depth / 1000 + 1;
                alpha = (depth % 1000) / 4;

            switch(level)
            {
                case(1):

                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, 0, alpha);
                    break;
                case(2):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, alpha, 255);
                    break;
                case(3):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, 255, 255-alpha);
                    break;
                case(4):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(alpha, 255, 0);
                    break;
                case(5):
                    depth_viz.at<Vec3b>(r, c) = Vec3b(255, 255-alpha, 0);
                    break;
                default:
                    depth_viz.at<Vec3b>(r, c) = Vec3b(0, 0, 0);
                    break;
           }

        }
   }
}



void Grabber::Exit()
{
	delete depth_stream_;
    delete color_stream_;

    device_->close();
    delete device_;

    openni::OpenNI::shutdown();
    ros::shutdown();
}

void Grabber::getImageCoordinate(Point3f& world_cord, Point& image_cord)
{
    image_cord.x = (int)(world_cord.x * fx / world_cord.z + cx);
    image_cord.y = (int)(world_cord.y * fy / world_cord.z + cy);

}

void Grabber::getWorldCoordinate(Point2f& image_cord, Point3f& cord)
{
    
    if(!color_mat.empty() && !depth_mat.empty() && image_cord.x < sizeColor.width && image_cord.y < sizeColor.height)
    {
        uint16_t d = depth_mat.at<uint16_t>(image_cord);
        cord.z = float(d) * 0.001f;
        cord.x = ((image_cord.x - cx) * cord.z) / fx;
        cord.y = ((image_cord.y - cy) * cord.z) / fy;
    }
}

void Grabber::getAllMarkerCoordinate(vector<vector<Point2f>>& corners, vector<int>& ids, vector<Point2f>& marker_center, vector<Point3f>& world_cord)
{
    for (int i = 0; i < ids.size(); i++)
    {
        Point2f center(0.f, 0.f);
        for (int j = 0; j < corners[i].size(); j++)
        {
            center += corners[i][j];
        }
        center /= 4.0;
        //cout << center.x << "," << center.y << endl;
        marker_center.push_back(center);
        Point3f cord3(0.f, 0.f, 0.f);
        getWorldCoordinate(marker_center[i], cord3);
        world_cord.push_back(cord3);
    }
}

void Grabber::sendMarkerTf(vector<Point3f>& marker_position, vector<int>& ids)
{
    static tf::TransformBroadcaster marker_position_broadcaster;
    for(int i = 0; i < marker_position.size(); i++)
    {
        tf::Transform transform;
        transform.setOrigin(tf::Vector3(marker_position[i].x, marker_position[i].y, marker_position[i].z));
        tf::Quaternion q;
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        ostringstream oss;
        oss << "marker_" << ids[i];
        marker_position_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_base", oss.str()));
    }
}

void Grabber::sendMarkerTf(vector<Vec3d>& marker_trans, vector<Vec3d>& marker_rot, vector<int>& ids)
{
    Mat rot(3, 3, CV_64FC1);
    Mat rot_to_ros(3, 3, CV_64FC1);
    rot_to_ros.at<double>(0,0) = -1.0;
    rot_to_ros.at<double>(0,1) = 0.0;
    rot_to_ros.at<double>(0,2) = 0.0;
    rot_to_ros.at<double>(1,0) = 0.0;
    rot_to_ros.at<double>(1,1) = 0.0;
    rot_to_ros.at<double>(1,2) = 1.0;
    rot_to_ros.at<double>(2,0) = 0.0;
    rot_to_ros.at<double>(2,1) = 1.0;
    rot_to_ros.at<double>(2,2) = 0.0;
    
    static tf::TransformBroadcaster marker_position_broadcaster;
    for(int i = 0; i < ids.size(); i++)
    {
        //tf::Transform transform;
        //transform.setOrigin(tf::Vector3(marker_trans[i][0], marker_trans[i][1], marker_trans[i][2]));
        //tf::Quaternion q;

        cv::Rodrigues(marker_rot[i], rot);
        rot.convertTo(rot, CV_64FC1);
/***
        double pitch = -atan2(rot_mat.at<double>(2,0), rot_mat.at<double>(2,1));
        double yaw   = acos(rot_mat.at<double>(2,2));
        double roll  = -atan2(rot_mat.at<double>(0,2), rot_mat.at<double>(1,2));

        //q.setRPY(marker_rot[i][0], marker_rot[i][1], marker_rot[i][2]);
        q.setRPY(roll, pitch, yaw);
        //q = q.inverse();
        transform.setRotation(q);
***/
        //rot = rot*rot_to_ros.t();
        
        
        tf::Matrix3x3 tf_rot(rot.at<double>(0,0), rot.at<double>(0,1), rot.at<double>(0,2),
                             rot.at<double>(1,0), rot.at<double>(1,1), rot.at<double>(1,2),
                             rot.at<double>(2,0), rot.at<double>(2,1), rot.at<double>(2,2));

        tf::Vector3 tf_trans(marker_trans[i][0], marker_trans[i][1], marker_trans[i][2]);
        tf::Transform transform(tf_rot, tf_trans);
        ostringstream oss;
        oss << "marker_" << ids[i];
        marker_position_broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_base", oss.str()));
    }
}
void Grabber::loadRobotPoseFile(string filename)
{
    ifstream inStream(filename);
    if (inStream)
    {
        vector<double> solution;
        int i = 0;
        while(!inStream.eof())
        {
            double in;
            inStream >> in;
            solution.push_back(in);
            i++;
        }
        vector<double>::iterator it = solution.end() - 1;
        solution.erase(it);
        for(int i = 0; i < solution.size(); i++)
        {
            cout << solution[i] << endl;
        }
        if(solution.size() != 10)
        {
            ROS_ERROR("Solution file invalid!");
            return;
        }
        robot_pose_tansform.setOrigin(tf::Vector3(solution[0] + solution[7], solution[1] + solution[8], solution[2] + solution[9]));
        //robot_pose_tansform.setOrigin(-tf::Vector3(solution[0], solution[1], solution[2]));
        tf::Quaternion q(solution[3], solution[4], solution[5], solution[6]);
        //q = q.inverse();
        robot_pose_tansform.setRotation(q);

        cout << "x: " << robot_pose_tansform.getRotation().x() << endl;
        cout << "y: " << robot_pose_tansform.getRotation().y() << endl;
        cout << "z: " << robot_pose_tansform.getRotation().z() << endl;
        cout << "w: " << robot_pose_tansform.getRotation().w() << endl;

    }
}
void Grabber::linkToRobotTf()
{
    static tf::TransformBroadcaster robot_pose_broadcaster;
    robot_pose_broadcaster.sendTransform(tf::StampedTransform(robot_pose_tansform, ros::Time::now(), "marker_0", "base_link"));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "openni2_aruco_tracker");
    
    bool isCalibrationMode = true;
    if(argc > 1)
    {
        printf("arg 1:%s\n", argv[1]);
        string arg = argv[1];
        if (arg == "false")
        {
            isCalibrationMode = false;
            ROS_INFO("calibrationMode disabled\n");
        }

    }
    
    Grabber grabber(isCalibrationMode);
    grabber.InitOpenNI();
    grabber.InitDevice();
    grabber.InitCalibration();
    grabber.InitDepthStream();
    grabber.InitColorStream();
    grabber.Run();
	grabber.Exit();

    return 0;
}
