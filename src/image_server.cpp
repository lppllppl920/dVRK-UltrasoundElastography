#include <stdio.h>
#include <string.h>    //strlen
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h> //inet_addr
#include <netdb.h>
#include <pthread.h> //for threading , link with lpthread
#include <iostream>
#include <math.h>

#include "igtlOSUtil.h"
#include "igtlMessageHeader.h"
#include "igtlImageMessage.h"
#include "igtlImageMetaMessage.h"
#include "igtlLabelMetaMessage.h"
#include "igtlServerSocket.h"
#include "igtlStringMessage.h"
#include "igtlMUSMessage.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Elastography.h"

igtl::USMessage::Pointer imgMsg;

void *connection_handler(void *socket_desc);

int SendImage(int socket, const char* name, const char* filename) {

	//TODO: You cannot new an object before deleting the old one
	std::cerr << "Reading " << filename << "...";
	cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	cv::namedWindow("Sending image", cv::WINDOW_NORMAL);
	cv::imshow("Sending image", image);
	cv::waitKey(100);

	int size[] = { image.rows, image.cols, 3 };       // image dimension
	float spacing[] = { 1.0, 1.0, 1.0 };     // spacing (mm/pixel)
	int svsize[] = { image.rows, image.cols, 3 };       // sub-volume size
	int svoffset[] = { 0, 0, 0 };           // sub-volume offset
	int scalarType = igtl::ImageMessage::TYPE_UINT8;          // scalar type

	imgMsg->SetDimensions(size);
	imgMsg->SetSpacing(spacing);
	imgMsg->SetScalarType(scalarType);
	imgMsg->SetDeviceName(name);
	imgMsg->SetSubVolume(svsize, svoffset);
	imgMsg->AllocateScalars();
	std::cout << image.cols << " " << image.rows << std::endl;
	memcpy(imgMsg->GetScalarPointer(), (void *) image.data,
			image.cols * image.rows * 3);
	// Following line may be called in case of 16-, 32-, and 64-bit scalar types.
	// imgMsg->SetEndian(igtl::ImageMessage::ENDIAN_BIG);
	//------------------------------------------------------------
	// Set image data (See GetTestImage() bellow for the details)
	igtl::Matrix4x4 matrix;
	igtl::IdentityMatrix(matrix);
	imgMsg->SetMatrix(matrix);
	//------------------------------------------------------------
	// Pack (serialize) and send
	imgMsg->Pack();
	int write_bytes = 0;
	int sum = 0;
	std::cout << " Ideal image bytes: " << imgMsg->GetPackSize() << std::endl;

	while (sum < imgMsg->GetPackSize()) {
		write_bytes = write(socket,
				(void *) ((unsigned char*) imgMsg->GetPackPointer() + sum),
				imgMsg->GetPackSize() - sum);

		if (write_bytes == -1) {
			perror("sending error");
			break;
		}

		sum += write_bytes;
		std::cout << "Have write image bytes: " << sum << std::endl;
	}
	return 1;
}

int main(int argc, char *argv[]) {

    Elastography test(argc, argv);
//	imgMsg = igtl::USMessage::New();
//// -------------- Server ----------------
//	int socket_desc, new_socket, c, *new_sock;
//	struct sockaddr_in server, client;
//	char *message;
//
//	//Create socket
//	socket_desc = socket(AF_INET, SOCK_STREAM, 0);
//	if (socket_desc == -1) {
//		printf("Could not create socket");
//	}
//
//	//Prepare the sockaddr_in structure
//	server.sin_family = AF_INET;
//	server.sin_addr.s_addr = INADDR_ANY;
//	server.sin_port = htons(8080);
//
//	//Bind
//	if (bind(socket_desc, (struct sockaddr *) &server, sizeof(server)) < 0) {
//		puts("bind failed");
//		return 1;
//	}
//	puts("bind done");
//
//	//Listen
//	listen(socket_desc, 3);
//
//	//Accept and incoming connection
//	puts("Waiting for incoming connections...");
//	c = sizeof(struct sockaddr_in);
//	while ((new_socket = accept(socket_desc, (struct sockaddr *) &client,
//			(socklen_t*) &c))) {
//		puts("Connection accepted");
//		//Reply to the client
//		pthread_t sniffer_thread;
//		new_sock = (int*) malloc(sizeof(int) * 1);
//		*new_sock = new_socket;
//
//		if (pthread_create(&sniffer_thread, NULL, connection_handler,
//				(void*) new_sock) < 0) {
//			perror("could not create thread");
//			return 1;
//		}
//
//		//Now join the thread , so that we dont terminate before the thread
//		//pthread_join( sniffer_thread , NULL);
//		puts("Handler assigned");
//	}
//
//	if (new_socket < 0) {
//		perror("accept failed");
//		return 1;
//	}
//
//	cv::destroyAllWindows();
//	close(socket_desc);
//	return 0;
}

/*
 * This will handle connection for each client
 * */
void *connection_handler(void *socket_desc) {
	//Get the socket descriptor
	int sock = *(int*) socket_desc;
	int read_size;

	for (int i = 0; i < 10; i++) {
		SendImage(sock, "MD_US_2D",
				"/home/xingtong/Pictures/US/0.tiff");
		sleep(1);
	}
	cv::destroyAllWindows();
	//Free the socket pointer
	close(sock);
//    free(socket_desc);
	return 0;
}

