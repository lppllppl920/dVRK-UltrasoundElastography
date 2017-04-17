/*
 * main_IGTLclient.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */
#include "IGTLclient.h"

IGTL_client * pointer;
unsigned char* candidate_header;

int main(int argc, char **argv) {

	ros::init(argc, argv, "my_socket_client");
	printf("main thread: %lu \n", IGTL_client::getThreadId());
	pointer = new IGTL_client(argc, argv);
	pointer->run();

	while (ros::ok()) {
		sleep(1);
	}
	return 0;
}

