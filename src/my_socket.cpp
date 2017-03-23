/* For sockaddr_in */
#include <netinet/in.h>
/* For socket functions */
#include <sys/socket.h>
/* For fcntl */
#include <fcntl.h>

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>

#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include <iostream>
#include <math.h>
#include <cstdlib>
#include <cstring>

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <std_msgs/String.h>

#include <igtlMUSMessage.h>
#include <igtlOSUtil.h>
#include <igtlMessageHeader.h>
#include <igtlTransformMessage.h>
#include <igtlImageMessage.h>
#include <igtlServerSocket.h>
#include <igtlStatusMessage.h>
#include <igtlPositionMessage.h>

#if OpenIGTLink_PROTOCOL_VERSION >= 2
#include <igtlPointMessage.h>
#include <igtlStringMessage.h>
#include <igtlBindMessage.h>
#endif //OpenIGTLink_PROTOCOL_VERSION >= 2

#include "igtlMUSMessage.h"

#define MAX_LINE 20736000
int ReceiveTransform(evbuffer * buf, igtl::MessageHeader * header);
int ReceivePosition(evbuffer * buf, igtl::MessageHeader * header);
int ReceiveImage(evbuffer * buf, igtl::MessageHeader * header);
int ReceiveStatus(evbuffer * buf, igtl::MessageHeader * header);

#if OpenIGTLink_PROTOCOL_VERSION >= 2
int ReceivePoint(evbuffer * buf, igtl::MessageHeader * header);
int ReceiveString(evbuffer * buf, igtl::MessageHeader * header);
int ReceiveBind(evbuffer * buf, igtl::MessageHeader * header);
#endif //OpenIGTLink_PROTOCOL_VERSION >= 2

unsigned long getThreadId() {
    std::string threadId = boost::lexical_cast<std::string>(
            boost::this_thread::get_id());
    unsigned long threadNumber = 0;
    sscanf(threadId.c_str(), "%lx", &threadNumber);
    return threadNumber;
}

void readcb(struct bufferevent *bev, void *ctx) {

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
            std::cout << "r: " << r << " pack size: "
                    << headerMsg->GetPackSize() << std::endl;
            perror("No enough data");
            return;
        }

        if (r != headerMsg->GetPackSize()) {
            continue;
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
    else if(strcmp(headerMsg->GetDeviceType(), "BIND") == 0) {

    }
    else {
        // if the data type is unknown, skip reading.
        std::cerr << "Receiving unknown device type: "
                << headerMsg->GetDeviceType() << std::endl;
        evbuffer_remove(input, dummy, MAX_LINE);
    }

}

void eventcb(struct bufferevent *bev, short event, void *ctx) {

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

void socket_run() {
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
    bufferevent_setcb(bev, readcb, NULL, eventcb, NULL);

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

void RosSubPub() {
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

int main(int argc, char **argv) {
    ros::init(argc, argv, "my_socket_client");
    printf("main thread: %lu \n", getThreadId());
    boost::thread thr(boost::bind(socket_run));
    boost::thread thr2(boost::bind(RosSubPub));

    thr.join();
    thr2.join();
    ros::shutdown();

    return 0;
}

int ReceiveTransform(evbuffer * buf, igtl::MessageHeader * header) {
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

int ReceivePosition(evbuffer * buf, igtl::MessageHeader * header) {
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

int ReceiveImage(evbuffer * buf, igtl::MessageHeader * header) {
    std::cerr << "Receiving IMAGE data type." << std::endl;

    // Create a message buffer to receive transform data
    igtl::ImageMessage::Pointer imgMsg;
    imgMsg = igtl::ImageMessage::New();
    imgMsg->SetMessageHeader(header);
    imgMsg->AllocatePack();

    int read_bytes = 0;
    // Receive Image data from the socket
    int sum = 0;
    while (sum != imgMsg->GetPackBodySize()) {
        read_bytes = evbuffer_remove(buf,
                (void *) ((unsigned char*) imgMsg->GetPackBodyPointer() + sum),
                imgMsg->GetPackBodySize() - sum);
        sum += read_bytes;
        std::cout << "Have read image bytes: " << sum << " Ideal image bytes: "
                << imgMsg->GetPackBodySize() << std::endl;
    }
    // Deserialize the Image data
    // If you want to skip CRC check, call Unpack() without argument.
    int c = imgMsg->Unpack(1);
    // if CRC check is OK
    if (c & igtl::MessageHeader::UNPACK_BODY) {
        // Retrive the image data
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

        cv::Mat image_rgb(size[0], size[1], CV_8UC3);
        memcpy((void *)image_rgb.data, imgMsg->GetScalarPointer(),
                image_rgb.cols * image_rgb.rows * 3);

        cv::imshow("Received image", image_rgb);
        cv::waitKey(100);
        return 1;
    }
    cv::destroyAllWindows();
    return 0;

}

int ReceiveStatus(evbuffer * buf, igtl::MessageHeader * header) {

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
int ReceivePoint(evbuffer * buf, igtl::MessageHeader * header) {

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

int ReceiveString(evbuffer * buf, igtl::MessageHeader * header) {

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

int ReceiveBind(evbuffer * buf, igtl::MessageHeader * header) {

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
