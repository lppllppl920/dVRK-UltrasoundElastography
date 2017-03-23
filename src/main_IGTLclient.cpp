/*
 * main_IGTLclient.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */
#include "IGTLclient.h"

int main(int argc, char **argv) {

    ros::init(argc, argv, "my_socket_client");
    printf("main thread: %lu \n", IGTL_client::getThreadId());
    IGTL_client client;
    client.run();
    return 0;
}



