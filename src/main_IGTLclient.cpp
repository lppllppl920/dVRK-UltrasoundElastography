/*
 * main_IGTLclient.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */
#include "IGTLclient.h"

IGTL_client * pointer;

int main(int argc, char **argv) {

	ros::init(argc, argv, "my_socket_client");
#ifdef DEBUG_OUTPUT
	printf("main thread: %lu \n", IGTL_client::getThreadId());
#endif
	pointer = new IGTL_client(argc, argv);
	pointer->run();

	while (ros::ok()) {
		sleep(1);
	}
	return 0;
}

