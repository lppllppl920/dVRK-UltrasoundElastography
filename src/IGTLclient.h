/*
 * IGTLclient.h
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */

#ifndef MY_SOCKET_SRC_IGTLCLIENT_H_
#define MY_SOCKET_SRC_IGTLCLIENT_H_

/* For sockaddr_in */
#include <netinet/in.h>
/* For socket functions */
#include <sys/socket.h>
/* For fcntl */
#include <fcntl.h>
// For inet_addr
#include <arpa/inet.h>

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>

#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <time.h>

#include <MUSiiCTCPServer.h>
#include <MUSiiCIGTLUtil.h>

#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
#include <igtlMEIMessage.h>

#if OpenIGTLink_PROTOCOL_VERSION >= 2
#include <igtlPointMessage.h>
#include <igtlStringMessage.h>
#include <igtlBindMessage.h>
#endif //OpenIGTLink_PROTOCOL_VERSION >= 2

#include <X11/Xlib.h>

#include "igtlMUSMessage.h"
#include "Elastography.h"

#define MAX_LINE 100000000

#define SANITY_CHECK_BYTES 14
#define DEVICE_TYPE_BYTES 12

//#define DEBUG_OUTPUT

class IGTL_client {
public:
	IGTL_client(int argc, char **argv);
	virtual ~IGTL_client();
	void socket_run();
	void ros_run();
	void run();

	void chatterCallback(const std_msgs::String::ConstPtr& msg);
	static unsigned long getThreadId();

public:
	unsigned char* candidate_header_;
	igtl::MessageHeader::Pointer header_msg_;
	igtl::USMessage::Pointer us_msg_;
	cv::Mat us_img_;
	int receiving_state_; // RECEIVING_HEADER, RECEIVING_BODY, RECEIVING_COMPLETE
	int bytes_received_; // This should be cleared to zero whenever the state changes
	int us_image_count_;
	enum {
		RECEIVING_HEADER = 0, RECEIVING_BODY, SEARCHING_HEADER
	};

	bool continue_write_image_;

    int Angle_;
    int FPS_;
    int FocusDepth_;
    int FocusSpacing_;
    int FocusCount_;
    int ImageSize_;
    int LineDensity_;
    int NumComponents_;
    int ProbeAngle_;
    int ProbeID_;
    int SamplingFrequency_;
    int TransmitFrequency_;
    int Pitch_;
    int Radius_;
    int ReferenceCount_;
    int SteeringAngle_;
    int USDataType_;

    int size_[3];          // image dimension
    float spacing_[3];       // spacing (mm/pixel)
    int svsize_[3];        // sub-volume size
    int svoffset_[3];      // sub-volume offset
    int scalarType_;       // scalar type

    int RF_or_BMODE_; //0 for RF, 1 for BMODE
    boost::shared_ptr<Elastography> elastography_;

private:
	boost::shared_ptr<boost::thread> socket_thread_;
	boost::shared_ptr<boost::thread> ros_thread_;
	boost::shared_ptr<boost::thread> elastography_thread_;


};

extern IGTL_client * pointer;
extern unsigned char* candidate_header;

#endif /* MY_SOCKET_SRC_IGTLCLIENT_H_ */
