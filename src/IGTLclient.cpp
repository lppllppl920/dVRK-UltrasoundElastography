/*
 * IGTLclient.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */

#include "IGTLclient.h"

IGTL_client::IGTL_client() {

}

IGTL_client::~IGTL_client() {

    socket_thread->join();
    ros_thread->join();
    ros::shutdown();
}

void IGTL_client::run() {
    socket_thread = boost::make_shared<boost::thread>(
            boost::bind(&IGTL_client::socket_run, this));
    ros_thread = boost::make_shared<boost::thread>(
            boost::bind(&IGTL_client::ros_run, this));
}

unsigned long IGTL_client::getThreadId() {
    std::string threadId = boost::lexical_cast<std::string>(
            boost::this_thread::get_id());
    unsigned long threadNumber = 0;
    sscanf(threadId.c_str(), "%lx", &threadNumber);
    return threadNumber;
}

void IGTL_client::readcb(struct bufferevent *bev, void *ctx) {

    printf("readcb thread: %lu \n", getThreadId());
    struct evbuffer *input, *output;
    char *dummy;
    size_t n;
    int i;
    input = bufferevent_get_input(bev);
    output = bufferevent_get_output(bev);

    // Create a message buffer to receive header
    igtl::MessageHeader::Pointer headerMsg;
    headerMsg = igtl::MessageHeader::New();
    // Initialize receive buffer
    headerMsg->InitPack();

    while (1) {
        // Receive generic header from the socket
        int r = evbuffer_remove(input, headerMsg->GetPackPointer(),
                headerMsg->GetPackSize());

        if (r == 0) {
            perror("Zero byte read. Something went wrong.");
            return;
        }

        if (r != headerMsg->GetPackSize()) {
            std::cout << "Received message header size: " << r
                    << " Ideal message header size: "
                    << headerMsg->GetPackSize() << std::endl;
            perror("Didn't receive enough message header data");
            return;
        } else {
            break;
        }
    }

    // Deserialize the header
    headerMsg->Unpack(1);
    std::cout << "Device Type: " << headerMsg->GetDeviceType() << std::endl;

    // Check data type and receive data body
    if (strcmp(headerMsg->GetDeviceType(), "TRANSFORM") == 0) {
        ReceiveTransform(input, headerMsg);
    } else if (strcmp(headerMsg->GetDeviceType(), "POSITION") == 0) {
        ReceivePosition(input, headerMsg);
    } else if (strcmp(headerMsg->GetDeviceType(), "IMAGE") == 0) {
        ReceiveImage(input, headerMsg);
    } else if (strcmp(headerMsg->GetDeviceType(), "STATUS") == 0) {
        ReceiveStatus(input, headerMsg);
    }
#if OpenIGTLink_PROTOCOL_VERSION >= 2
    else if (strcmp(headerMsg->GetDeviceType(), "POINT") == 0) {
        ReceivePoint(input, headerMsg);
    } else if (strcmp(headerMsg->GetDeviceType(), "STRING") == 0) {
        ReceiveString(input, headerMsg);
    } else if (strcmp(headerMsg->GetDeviceType(), "BIND") == 0) {
        ReceiveBind(input, headerMsg);
    }
#endif //OpenIGTLink_PROTOCOL_VERSION >= 2
    else if (strcmp(headerMsg->GetDeviceName(), "US") == 0) {
        ReceiveUS(input, headerMsg);
    } else {
        // if the data type is unknown, skip reading.
        std::cerr << "Receiving unknown device type: "
                << headerMsg->GetDeviceType() << std::endl;
        evbuffer_remove(input, dummy, MAX_LINE);
    }

}

void IGTL_client::eventcb(struct bufferevent *bev, short event, void *ctx) {

    if (event & BEV_EVENT_CONNECTED) {
        /* We're connected to 127.0.0.1:8080.   Ordinarily we'd do
         something here, like start reading or writing. */
        printf("We are connected to server!\n");
        //TODO: This needs to be set according to the image size
        // The minimum value is how many bytes we need to have before invoking the read callback function
        bufferevent_setwatermark(bev, EV_READ, 1080 * 1920 * 3, MAX_LINE);
        bufferevent_enable(bev, EV_READ | EV_WRITE);
    } else if (event & BEV_EVENT_ERROR) {
        /* An error occured while connecting. */
        perror("Error occured while connecting the server!");
        bufferevent_free(bev);
    } else if (event & BEV_EVENT_TIMEOUT) {
        /* must be a timeout event handle, handle it */
        perror("Time out occured!");
    } else {
        perror("Event callback invoked");
    }

}

void IGTL_client::ros_run() {
    ros::NodeHandle n;

    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
    ros::Rate loop_rate(1);
    /**
     * A count of how many messages we have sent. This is used to create
     * a unique string for each message.
     */
    int count = 0;
    while (ros::ok()) {
        printf("ros thread: %lu \n", getThreadId());
        /**
         * This is a message object. You stuff it with data, and then publish it.
         */
        std_msgs::String msg;

        std::stringstream ss;
        ss << "hello world " << count;
        msg.data = ss.str();

        ROS_INFO("%s", msg.data.c_str());

        /**
         * The publish() function is how you send messages. The parameter
         * is the message object. The type of this object must agree with the type
         * given as a template parameter to the advertise<>() call, as was done
         * in the constructor above.
         */
        chatter_pub.publish(msg);

        ros::spinOnce();

        loop_rate.sleep();
        ++count;
    }
}

void IGTL_client::socket_run() {
    printf("run thread: %lu \n", getThreadId());
    struct event_base *base;
    struct bufferevent *bev;
    struct sockaddr_in sin;
    base = event_base_new();

    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = htonl(0x7f000001); /* 127.0.0.1 */
    sin.sin_port = htons(8080); /* Port 8080 */

    bev = bufferevent_socket_new(base, -1, BEV_OPT_CLOSE_ON_FREE);
    bufferevent_setcb(bev, IGTL_client::readcb, NULL, IGTL_client::eventcb,
            NULL);
    /* Note that you only get a BEV_EVENT_CONNECTED event if you launch the connect()
     * attempt using bufferevent_socket_connect(). If you call connect() on your own,
     * the connection gets reported as a write.
     */
    if (bufferevent_socket_connect(bev, (struct sockaddr *) &sin, sizeof(sin))
            < 0) {
        /* Error starting connection */
        bufferevent_free(bev);
        return;
    }

    event_base_dispatch(base);
    bufferevent_free(bev);
    return;
}
int IGTL_client::ReceiveUS(evbuffer * buf, igtl::MessageHeader * header) {
    std::cerr << "Receiving US IMAGE data type." << std::endl;

    cv::Mat image;
    igtl::USMessage::Pointer usImgMsg;
    usImgMsg = igtl::USMessage::New();
    usImgMsg->SetMessageHeader(header);
    usImgMsg->AllocatePack();

    int read_bytes = 0;
    // Receive Image data from the socket
    int sum = 0;
    while (true) {
        read_bytes =
                evbuffer_remove(buf,
                        (void *) ((unsigned char*) usImgMsg->GetPackBodyPointer()
                                + sum), usImgMsg->GetPackBodySize() - sum);
        if (read_bytes == -1 || read_bytes == 0) {
            perror("No enough data for US Image");
            return -1;
        }
        sum += read_bytes;
        if (usImgMsg->GetPackBodySize() == sum) {
            break;
        }
        std::cout << "Have read image bytes: " << sum << " Ideal image bytes: "
                << usImgMsg->GetPackBodySize() << std::endl;
    }

    // Deserialize the Image data
    int c = usImgMsg->Unpack(1);
    // if CRC check is OK
    if (c & igtl::MessageHeader::UNPACK_BODY) {
        cv::namedWindow("Received image", CV_WINDOW_NORMAL);
        int size[3];          // image dimension
        float spacing[3];       // spacing (mm/pixel)
        int svsize[3];        // sub-volume size
        int svoffset[3];      // sub-volume offset
        int scalarType;       // scalar type

        scalarType = usImgMsg->GetScalarType();
        usImgMsg->GetDimensions(size);
        usImgMsg->GetSpacing(spacing);
        usImgMsg->GetSubVolume(svsize, svoffset);
        int Angle = usImgMsg->GetExtensionAngle();
        int FPS = usImgMsg->GetFPS();
        int FocusDepth = usImgMsg->GetFocusDepth();
        int FocusSpacing = usImgMsg->GetFocusSpacing();
        int FocusCount = (int) usImgMsg->GetFocus_Count();
        int ImageSize = usImgMsg->GetImageSize();
        int LineDensity = usImgMsg->GetLineDensity();
        int NumComponents = usImgMsg->GetNumComponents();
        int ProbeAngle = usImgMsg->GetProbeAngle();
        int ProbeID = usImgMsg->GetProbeID();
        int SamplingFrequency = usImgMsg->GetSamplingFrequency();
        int TransmitFrequency = usImgMsg->GetTransmitFrequency();
        int Pitch = usImgMsg->GetPitch();
        int Radius = usImgMsg->GetRadius();
        int ReferenceCount = usImgMsg->GetReferenceCount();
        int SteeringAngle = usImgMsg->GetSteeringAngle();
        int USDataType = usImgMsg->GetUSDataType();

        std::cerr << "Device Name           : " << usImgMsg->GetDeviceName()
                << std::endl;
        std::cerr << "Scalar Type           : " << scalarType << std::endl;
        std::cerr << "Dimensions            : (" << size[0] << ", " << size[1]
                << ", " << size[2] << ")" << std::endl;
        std::cerr << "Spacing               : (" << spacing[0] << ", "
                << spacing[1] << ", " << spacing[2] << ")" << std::endl;
        std::cerr << "Sub-Volume dimensions : (" << svsize[0] << ", "
                << svsize[1] << ", " << svsize[2] << ")" << std::endl;
        std::cerr << "Sub-Volume offset     : (" << svoffset[0] << ", "
                << svoffset[1] << ", " << svoffset[2] << ")" << std::endl;
        std::cerr << "Angle                 : " << Angle << std::endl;
        std::cerr << "FPS                   : " << FPS << std::endl;
        std::cerr << "FocusDepth            : " << FocusDepth << std::endl;
        std::cerr << "FocusSpacing          : " << FocusSpacing << std::endl;
        std::cerr << "FocusCount            : " << FocusCount << std::endl;
        std::cerr << "ImageSize             : " << ImageSize << std::endl;
        std::cerr << "LineDensity           : " << LineDensity << std::endl;
        std::cerr << "NumComponents         : " << NumComponents << std::endl;
        std::cerr << "ProbeAngle            : " << ProbeAngle << std::endl;
        std::cerr << "ProbeID               : " << ProbeID << std::endl;
        std::cerr << "SamplingFrequency     : " << SamplingFrequency
                << std::endl;
        std::cerr << "TransmitFrequency     : " << TransmitFrequency
                << std::endl;
        std::cerr << "Pitch                 : " << Pitch << std::endl;
        std::cerr << "Radius                : " << Radius << std::endl;
        std::cerr << "ReferenceCount        : " << ReferenceCount << std::endl;
        std::cerr << "SteeringAngle         : " << SteeringAngle << std::endl;
        std::cerr << "USDataType            : " << USDataType << std::endl;

        if (scalarType == igtl::ImageMessage::TYPE_UINT8) {
            image = cv::Mat::zeros(size[0], size[1], CV_8UC(size[2]));
        } else if (scalarType == igtl::ImageMessage::TYPE_INT8) {
            image = cv::Mat::zeros(size[0], size[1], CV_8SC(size[2]));
        } else if (scalarType == igtl::ImageMessage::TYPE_INT16) {
            image = cv::Mat::zeros(size[0], size[1], CV_16SC(size[2]));
        } else if (scalarType == igtl::ImageMessage::TYPE_UINT16) {
            image = cv::Mat::zeros(size[0], size[1], CV_16UC(size[2]));
        } else if (scalarType == igtl::ImageMessage::TYPE_INT32) {
            image = cv::Mat::zeros(size[0], size[1], CV_32SC(size[2]));
        } else if (scalarType == igtl::ImageMessage::TYPE_FLOAT32) {
            image = cv::Mat::zeros(size[0], size[1], CV_32FC(size[2]));
        } else if (scalarType == igtl::ImageMessage::TYPE_FLOAT64) {
            image = cv::Mat::zeros(size[0], size[1], CV_64FC(size[2]));
        } else {
            perror("No supported type found");
            return -1;
        }

        if (NumComponents == 1) {
            memcpy((void *) image.data, usImgMsg->GetScalarPointer(),
                    usImgMsg->GetImageSize());
            cv::imshow("Received image", image);
            cv::waitKey(100);
        }

    }

}

int IGTL_client::ReceiveImage(evbuffer * buf, igtl::MessageHeader * header) {
    std::cerr << "Receiving IMAGE data type." << std::endl;

    // Create a message buffer to receive image data
    igtl::ImageMessage::Pointer imgMsg;
    imgMsg = igtl::ImageMessage::New();
    imgMsg->SetMessageHeader(header);
    imgMsg->AllocatePack();

    int read_bytes = 0;
    // Receive Image data from the socket
    int sum = 0;
    while (true) {
        read_bytes = evbuffer_remove(buf,
                (void *) ((unsigned char*) imgMsg->GetPackBodyPointer() + sum),
                imgMsg->GetPackBodySize() - sum);

        if (read_bytes == -1 || read_bytes == 0) {
            perror("No enough data for US Image");
            return -1;
        }

        sum += read_bytes;
        if (imgMsg->GetPackBodySize() == sum) {
            break;
        }

        std::cout << "Have read image bytes: " << sum << " Ideal image bytes: "
                << imgMsg->GetPackBodySize() << std::endl;
    }

    // Deserialize the Image data
    int c = imgMsg->Unpack(1);
    // if CRC check is OK
    if (c & igtl::MessageHeader::UNPACK_BODY) {
        // Retrive the image data
        cv::namedWindow("Received image", CV_WINDOW_NORMAL);
        int size[3];          // image dimension
        float spacing[3];       // spacing (mm/pixel)
        int svsize[3];        // sub-volume size
        int svoffset[3];      // sub-volume offset
        int scalarType;       // scalar type

        scalarType = imgMsg->GetScalarType();
        imgMsg->GetDimensions(size);
        imgMsg->GetSpacing(spacing);
        imgMsg->GetSubVolume(svsize, svoffset);

        std::cerr << "Device Name           : " << imgMsg->GetDeviceName()
                << std::endl;
        std::cerr << "Scalar Type           : " << scalarType << std::endl;
        std::cerr << "Dimensions            : (" << size[0] << ", " << size[1]
                << ", " << size[2] << ")" << std::endl;
        std::cerr << "Spacing               : (" << spacing[0] << ", "
                << spacing[1] << ", " << spacing[2] << ")" << std::endl;
        std::cerr << "Sub-Volume dimensions : (" << svsize[0] << ", "
                << svsize[1] << ", " << svsize[2] << ")" << std::endl;
        std::cerr << "Sub-Volume offset     : (" << svoffset[0] << ", "
                << svoffset[1] << ", " << svoffset[2] << ")" << std::endl;

        if (size[2] == 3) {
            cv::Mat image_rgb(size[0], size[1], CV_8UC3);
            memcpy((void *) image_rgb.data, imgMsg->GetScalarPointer(),
                    size[0] * size[1] * size[2]);
            cv::imshow("Received image", image_rgb);
            cv::waitKey(100);

        } else if (size[2] == 1) {
            cv::Mat image_gray(size[0], size[1], CV_8UC1);
            memcpy((void *) image_gray.data, imgMsg->GetScalarPointer(),
                    size[0] * size[1] * size[2]);
            cv::imshow("Received image", image_gray);
            cv::waitKey(100);
        }
        return 1;
    }

    cv::destroyAllWindows();
    return 0;

}

int IGTL_client::ReceiveTransform(evbuffer * buf,
        igtl::MessageHeader * header) {
    std::cerr << "Receiving TRANSFORM data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::TransformMessage::Pointer transMsg;
    transMsg = igtl::TransformMessage::New();
    transMsg->SetMessageHeader(header);
    transMsg->AllocatePack();

    // Receive transform data from the socket
    evbuffer_remove(buf, transMsg->GetPackBodyPointer(),
            transMsg->GetPackBodySize());

    // Deserialize the transform data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = transMsg->Unpack(1);

    if (c & igtl::MessageHeader::UNPACK_BODY) // if CRC check is OK
            {
        // Retrive the transform data
        igtl::Matrix4x4 matrix;
        transMsg->GetMatrix(matrix);
        igtl::PrintMatrix(matrix);
        return 1;
    }

    return 0;

}

int IGTL_client::ReceivePosition(evbuffer * buf, igtl::MessageHeader * header) {
    std::cerr << "Receiving POSITION data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::PositionMessage::Pointer positionMsg;
    positionMsg = igtl::PositionMessage::New();
    positionMsg->SetMessageHeader(header);
    positionMsg->AllocatePack();

    // Receive position position data from the socket
    evbuffer_remove(buf, positionMsg->GetPackBodyPointer(),
            positionMsg->GetPackBodySize());

    // Deserialize the transform data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = positionMsg->Unpack(1);
    // if CRC check is OK
    if (c & igtl::MessageHeader::UNPACK_BODY) {
        // Retrive the transform data
        float position[3];
        float quaternion[4];

        positionMsg->GetPosition(position);
        positionMsg->GetQuaternion(quaternion);

        std::cerr << "position   = (" << position[0] << ", " << position[1]
                << ", " << position[2] << ")" << std::endl;
        std::cerr << "quaternion = (" << quaternion[0] << ", " << quaternion[1]
                << ", " << quaternion[2] << ", " << quaternion[3] << ")"
                << std::endl << std::endl;

        return 1;
    }
    return 0;

}

int IGTL_client::ReceiveStatus(evbuffer * buf, igtl::MessageHeader * header) {

    std::cerr << "Receiving STATUS data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::StatusMessage::Pointer statusMsg;
    statusMsg = igtl::StatusMessage::New();
    statusMsg->SetMessageHeader(header);
    statusMsg->AllocatePack();

    // Receive status data from the socket
    evbuffer_remove(buf, statusMsg->GetPackBodyPointer(),
            statusMsg->GetPackBodySize());
    // Deserialize the status data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = statusMsg->Unpack(1);

    if (c & igtl::MessageHeader::UNPACK_BODY) // if CRC check is OK
            {
        std::cerr << "========== STATUS ==========" << std::endl;
        std::cerr << " Code      : " << statusMsg->GetCode() << std::endl;
        std::cerr << " SubCode   : " << statusMsg->GetSubCode() << std::endl;
        std::cerr << " Error Name: " << statusMsg->GetErrorName() << std::endl;
        std::cerr << " Status    : " << statusMsg->GetStatusString()
                << std::endl;
        std::cerr << "============================" << std::endl;
    }

    return 0;

}

#if OpenIGTLink_PROTOCOL_VERSION >= 2
int IGTL_client::ReceivePoint(evbuffer * buf, igtl::MessageHeader * header) {

    std::cerr << "Receiving POINT data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::PointMessage::Pointer pointMsg;
    pointMsg = igtl::PointMessage::New();
    pointMsg->SetMessageHeader(header);
    pointMsg->AllocatePack();

    // Receive point data from the socket
    evbuffer_remove(buf, pointMsg->GetPackBodyPointer(),
            pointMsg->GetPackBodySize());
    // Deserialize the point data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = pointMsg->Unpack(1);

    if (c & igtl::MessageHeader::UNPACK_BODY) // if CRC check is OK
            {
        int nElements = pointMsg->GetNumberOfPointElement();
        for (int i = 0; i < nElements; i++) {
            igtl::PointElement::Pointer pointElement;
            pointMsg->GetPointElement(i, pointElement);

            igtlUint8 rgba[4];
            pointElement->GetRGBA(rgba);

            igtlFloat32 pos[3];
            pointElement->GetPosition(pos);

            std::cerr << "========== Element #" << i << " =========="
                    << std::endl;
            std::cerr << " Name      : " << pointElement->GetName()
                    << std::endl;
            std::cerr << " GroupName : " << pointElement->GetGroupName()
                    << std::endl;
            std::cerr << " RGBA      : ( " << (int) rgba[0] << ", "
                    << (int) rgba[1] << ", " << (int) rgba[2] << ", "
                    << (int) rgba[3] << " )" << std::endl;
            std::cerr << " Position  : ( " << std::fixed << pos[0] << ", "
                    << pos[1] << ", " << pos[2] << " )" << std::endl;
            std::cerr << " Radius    : " << std::fixed
                    << pointElement->GetRadius() << std::endl;
            std::cerr << " Owner     : " << pointElement->GetOwner()
                    << std::endl;
            std::cerr << "================================" << std::endl;
        }
    }

    return 1;
}

int IGTL_client::ReceiveString(evbuffer * buf, igtl::MessageHeader * header) {

    std::cerr << "Receiving STRING data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::StringMessage::Pointer stringMsg;
    stringMsg = igtl::StringMessage::New();
    stringMsg->SetMessageHeader(header);
    stringMsg->AllocatePack();

    // Receive string data from the socket
    evbuffer_remove(buf, stringMsg->GetPackBodyPointer(),
            stringMsg->GetPackBodySize());
    // Deserialize the string data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = stringMsg->Unpack(1);

    if (c & igtl::MessageHeader::UNPACK_BODY) // if CRC check is OK
            {
        std::cerr << "Encoding: " << stringMsg->GetEncoding() << "; "
                << "String: " << stringMsg->GetString() << std::endl;
    }

    return 1;
}

int IGTL_client::ReceiveBind(evbuffer * buf, igtl::MessageHeader * header) {

    std::cerr << "Receiving BIND data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::BindMessage::Pointer bindMsg;
    bindMsg = igtl::BindMessage::New();
    bindMsg->SetMessageHeader(header);
    bindMsg->AllocatePack();

    // Receive bind data from the socket
    evbuffer_remove(buf, bindMsg->GetPackBodyPointer(),
            bindMsg->GetPackBodySize());
    // Deserialize the bind data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = bindMsg->Unpack(1);

    if (c & igtl::MessageHeader::UNPACK_BODY) // if CRC check is OK
            {
        int n = bindMsg->GetNumberOfChildMessages();

        for (int i = 0; i < n; i++) {
            if (strcmp(bindMsg->GetChildMessageType(i), "STRING") == 0) {
                igtl::StringMessage::Pointer stringMsg;
                stringMsg = igtl::StringMessage::New();
                bindMsg->GetChildMessage(i, stringMsg);
                stringMsg->Unpack(0);
                std::cerr << "Message type: STRING" << std::endl;
                std::cerr << "Message name: " << stringMsg->GetDeviceName()
                        << std::endl;
                std::cerr << "Encoding: " << stringMsg->GetEncoding() << "; "
                        << "String: " << stringMsg->GetString() << std::endl;
            } else if (strcmp(bindMsg->GetChildMessageType(i), "TRANSFORM")
                    == 0) {
                igtl::TransformMessage::Pointer transMsg;
                transMsg = igtl::TransformMessage::New();
                bindMsg->GetChildMessage(i, transMsg);
                transMsg->Unpack(0);
                std::cerr << "Message type: TRANSFORM" << std::endl;
                std::cerr << "Message name: " << transMsg->GetDeviceName()
                        << std::endl;
                igtl::Matrix4x4 matrix;
                transMsg->GetMatrix(matrix);
                igtl::PrintMatrix(matrix);
            }
        }
    }

    return 1;
}
#endif //OpenIGTLink_PROTOCOL_VERSION >= 2
