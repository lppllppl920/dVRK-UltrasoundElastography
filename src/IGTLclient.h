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

#include <boost/bind.hpp>
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
#define MAX_LINE 100000000

class IGTL_client {
public:
    IGTL_client();
    virtual ~IGTL_client();
    void socket_run();
    void ros_run();
    void run();

    static unsigned long getThreadId();
    static void readcb(struct bufferevent *bev, void *ctx);
    static void eventcb(struct bufferevent *bev, short event, void *ctx);
    static int ReceiveTransform(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceivePosition(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceiveImage(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceiveStatus(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceivePoint(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceiveString(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceiveBind(evbuffer * buf, igtl::MessageHeader * header);
    static int ReceiveUS(evbuffer * buf, igtl::MessageHeader * header);

private:
    boost::shared_ptr<boost::thread> socket_thread;
    boost::shared_ptr<boost::thread> ros_thread;
};

#endif /* MY_SOCKET_SRC_IGTLCLIENT_H_ */
