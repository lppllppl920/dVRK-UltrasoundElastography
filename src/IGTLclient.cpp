/*
 * IGTLclient.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */

#include "IGTLclient.h"

IGTL_client::IGTL_client(int argc, char **argv) {
	receiving_state_ = SEARCHING_HEADER;
	bytes_received_ = 0;

	header_msg_ = igtl::MessageHeader::New();
	us_msg_ = igtl::USMessage::New();
	us_image_count_ = 0;

    if (argc != 11) {
        std::cout << "number of parameters is not right\n Parameters list: \n" <<
                "Strain or displacement image(0/1), Number of frames for single elastogram(int), GPU Device ID(int)\n" <<
                "Number of threads assigned for elastography(int), Number of new frames of RF data added for a new elastogram(int)\n" <<
                "Window size of Normalized cross correlation(NCC) for elastography(int), overlap for NCC(float)\n" <<
                "Overall displacement(float), RF or BMode Image(0/1), Ultrasound machine IP address(character array)\n" <<
                "example: 1 8 0 8 2 20 1.0 5.0 0 10.162.34.81\n";
        exit(1);
    }

	RF_or_BMODE_ = atoi(argv[9]);
	us_ip_address_ = argv[10];

#ifdef DEBUG_OUTPUT
	printf("US IP: %s", us_ip_address_);
#endif

	elastography_ = boost::make_shared<Elastography>(argc - 2, argv);
	continue_write_image_ = false;
	candidate_header_ = NULL;
}

IGTL_client::~IGTL_client() {

	cv::destroyAllWindows();
	if(RF_or_BMODE_ == 0) {
	    elastography_thread_->join();
	}
	socket_thread_->join();
	// TODO: uncomment this line if you need to use ros publisher and subscriber
//	ros_thread_->join();
	ros::shutdown();
}


void IGTL_client::run() {
    XInitThreads();
    // TODO: uncomment this line if you need to use ros publisher and subscriber
//    ros_thread_ = boost::make_shared < boost::thread
//            > (boost::bind(&IGTL_client::ros_run, this));
	socket_thread_ = boost::make_shared < boost::thread
			> (boost::bind(&IGTL_client::socket_run, this));
	if(RF_or_BMODE_ == 0) {
	    elastography_thread_ = boost::make_shared<boost::thread>
	    (boost::bind(&Elastography::CalculateElastography, elastography_));
	}
}

unsigned long IGTL_client::getThreadId() {
	std::string threadId = boost::lexical_cast < std::string
			> (boost::this_thread::get_id());
	unsigned long threadNumber = 0;
	sscanf(threadId.c_str(), "%lx", &threadNumber);
	return threadNumber;
}

void readcb_global(struct bufferevent *bev, void *ctx) {
#ifdef DEBUG_OUTPUT
    printf("readcb thread: %lu \n", pointer->getThreadId());
#endif

    struct evbuffer *input, *output;
	char *dummy;
	char Device_type[DEVICE_TYPE_BYTES];
	unsigned char temp[SANITY_CHECK_BYTES];
	input = bufferevent_get_input(bev);
	output = bufferevent_get_output(bev);

	// To speed up the search, we can first check the first two bytes and following 12 bytes to make sure the version number and type is right
	// which means possible version number, then go unpack and see if we get a right header
	if (pointer->receiving_state_ == IGTL_client::SEARCHING_HEADER) {
#ifdef DEBUG_OUTPUT
	    std::cout << "Searching header..\n";
#endif
		// Initialize candidate header
		if (pointer->bytes_received_ == 0) {
			pointer->candidate_header_ = (unsigned char *) malloc(SANITY_CHECK_BYTES);
			pointer->header_msg_->InitPack();
		}

#ifdef DEBUG_OUTPUT
		assert(SANITY_CHECK_BYTES > pointer->bytes_received_);
#endif
		int r = evbuffer_remove(input,
				(void *) (pointer->candidate_header_ + pointer->bytes_received_),
				(SANITY_CHECK_BYTES - pointer->bytes_received_));

		if (r > 0) {

			pointer->bytes_received_ += r;
			if (pointer->bytes_received_ >= SANITY_CHECK_BYTES) {
				// Let's do the sanity check
				unsigned short V = (unsigned short) (pointer->candidate_header_[0] << 8)
						| (unsigned short) pointer->candidate_header_[1];
				memcpy(Device_type, (void *) (pointer->candidate_header_ + 2),
						DEVICE_TYPE_BYTES);
				if ((V == 1 || V == 2) && (strcmp(Device_type, "IMAGE") == 0)) {
#ifdef DEBUG_OUTPUT
					std::cout << "Valid header found! \n";
#endif
					pointer->receiving_state_ = IGTL_client::RECEIVING_HEADER;
					memcpy(pointer->header_msg_->GetPackPointer(),
							(void *) pointer->candidate_header_, SANITY_CHECK_BYTES);
					pointer->bytes_received_ = SANITY_CHECK_BYTES;

				} else {

					// Search the entire input buffer to find the valid header
					while (true) {
						// Invalid header, need to search for another one
						memcpy((void *) temp, (void *) (pointer->candidate_header_ + 1),
								(SANITY_CHECK_BYTES - 1));
						r = evbuffer_remove(input,
								(void *) (temp + SANITY_CHECK_BYTES - 1), 1);
						if (r <= 0) {
							memcpy(pointer->candidate_header_, temp,
									SANITY_CHECK_BYTES - 1);
							pointer->receiving_state_ =
									IGTL_client::SEARCHING_HEADER;
							pointer->bytes_received_ = SANITY_CHECK_BYTES - 1;
							return;

						} else {
							memcpy(pointer->candidate_header_, temp, SANITY_CHECK_BYTES);
							V = (unsigned short) (pointer->candidate_header_[0] << 8)
									| (unsigned short) pointer->candidate_header_[1];
							memcpy(Device_type, (void *) (pointer->candidate_header_ + 2),
									DEVICE_TYPE_BYTES);
							if ((V == 1 || V == 2)
									&& (strcmp(Device_type, "IMAGE") == 0)) {
#ifdef DEBUG_OUTPUT
								std::cout << "Valid header found! \n";
#endif
								pointer->receiving_state_ =
										IGTL_client::RECEIVING_HEADER;
								memcpy(pointer->header_msg_->GetPackPointer(),
										(void *) pointer->candidate_header_,
										SANITY_CHECK_BYTES);
								pointer->bytes_received_ = SANITY_CHECK_BYTES;
								break;
							}
						}
					}

				}

			} else {
				// Not enough bytes, continue for more data
				pointer->receiving_state_ = IGTL_client::SEARCHING_HEADER;
				return;
			}

		} else {
			perror("zero byte received, waiting to receive more data to complete a header sanity check\n");
			return;
		}

	}

	// Receiving header
	if (pointer->receiving_state_ == IGTL_client::RECEIVING_HEADER) {
#ifdef DEBUG_OUTPUT
	    std::cout << "Receiving header..\n";
#endif
		// If we start to receive a new header, first initialize the HeaderMessage object
		if (pointer->bytes_received_ == 0) {
			pointer->header_msg_->InitPack();
		}

#ifdef DEBUG_OUTPUT
		assert(pointer->header_msg_->GetPackSize() > pointer->bytes_received_);
#endif
		// Receive generic header from the socket
		int r =
				evbuffer_remove(input,
						(void*) ((unsigned char *) pointer->header_msg_->GetPackPointer()
								+ pointer->bytes_received_),
						pointer->header_msg_->GetPackSize()
								- pointer->bytes_received_);

		if (r <= 0) {
			perror("zero byte received, waiting to receive more data to complete a message header\n");
			return;

		} else {

			pointer->bytes_received_ += r;
#ifdef DEBUG_OUTPUT
			std::cout << "Have read header bytes: " << pointer->bytes_received_
					<< " Ideal header bytes: "
					<< pointer->header_msg_->GetPackSize() << std::endl;
#endif
			// Header receiving complete
			if (pointer->bytes_received_
					>= pointer->header_msg_->GetPackSize()) {

				// Deserialize the header
				int c = pointer->header_msg_->Unpack(1);
				// If the header is
				if (c & igtl::MessageHeader::UNPACK_HEADER) {
#ifdef DEBUG_OUTPUT
					std::cout << "Device Type: "
							<< pointer->header_msg_->GetDeviceType()
							<< " Device Name: "
							<< pointer->header_msg_->GetDeviceName()
							<< std::endl;
#endif
					// US Message received
					if (strcmp(pointer->header_msg_->GetDeviceName(), "MD_US_2D")
							== 0) {
						pointer->receiving_state_ = IGTL_client::RECEIVING_BODY;
						pointer->bytes_received_ = 0;
					} else if (strcmp(pointer->header_msg_->GetDeviceName(), "MD_BMODE_2D") == 0){
                        pointer->receiving_state_ = IGTL_client::RECEIVING_BODY;
                        pointer->bytes_received_ = 0;
					} else {
                        // Didn't receive the desired type of message, skip it and research for desired header
                        perror("Device Name is not MD_US_2D or MD_BMODE_2D\n");
                        pointer->receiving_state_ =
                                IGTL_client::SEARCHING_HEADER;
                        evbuffer_remove(input, dummy, MAX_LINE);
                        pointer->bytes_received_ = 0;
                        free(pointer->candidate_header_);
                        return;
					}
				} else {
					// If the header unpacking failed, which means we didn't find a valid header
					// We need to search for header again
					//TODO:
					pointer->receiving_state_ = IGTL_client::SEARCHING_HEADER;
					evbuffer_remove(input, dummy, MAX_LINE);
					pointer->bytes_received_ = 0;
					if(pointer->candidate_header_ != NULL) {
					    free(pointer->candidate_header_);
					}
					return;
				}

			} else {
				// Haven't received enough header data
				// Continue with it for the next readcb
				pointer->receiving_state_ = IGTL_client::RECEIVING_HEADER;
				bufferevent_setwatermark(bev, EV_READ,
						pointer->header_msg_->GetPackSize()
								- pointer->bytes_received_,
						pointer->header_msg_->GetPackSize()
								- pointer->bytes_received_);
				return;
			}
		}
	}

	// Receiving body
	if (pointer->receiving_state_ == IGTL_client::RECEIVING_BODY) {
#ifdef DEBUG_OUTPUT
	    std::cout << "Receiving Body..\n";
#endif
		// Initialize US Message when starting to receive body part
		if (pointer->bytes_received_ == 0) {
			pointer->us_msg_->SetMessageHeader(pointer->header_msg_);
			pointer->us_msg_->AllocatePack();
		}

#ifdef DEBUG_OUTPUT
		assert(pointer->us_msg_->GetPackBodySize() > pointer->bytes_received_);
#endif
		int r =
				evbuffer_remove(input,
						(void *) ((unsigned char*) pointer->us_msg_->GetPackBodyPointer()
								+ pointer->bytes_received_),
						pointer->us_msg_->GetPackBodySize()
								- pointer->bytes_received_);

		if (r <= 0) {
			perror("zero byte received, waiting to receive more data to complete the US body part\n");
			return;
		} else {

			pointer->bytes_received_ += r;
#ifdef DEBUG_OUTPUT
			std::cout << "Have read image bytes: " << pointer->bytes_received_
					<< " Ideal image bytes: "
					<< pointer->us_msg_->GetPackBodySize() << std::endl;
#endif
			if (pointer->bytes_received_
					>= pointer->us_msg_->GetPackBodySize()) {
				// A complete US image received
				pointer->receiving_state_ = IGTL_client::RECEIVING_HEADER;
				pointer->bytes_received_ = 0;
				int c = pointer->us_msg_->Unpack(1);

				if (c & igtl::MessageHeader::UNPACK_BODY) {

					cv::namedWindow("Received image", CV_WINDOW_FULLSCREEN);
					pointer->scalarType_ = pointer->us_msg_->GetScalarType();
					pointer->us_msg_->GetDimensions(pointer->size_);
					pointer->us_msg_->GetSpacing(pointer->spacing_);
					pointer->us_msg_->GetSubVolume(pointer->svsize_, pointer->svoffset_);
					pointer->Angle_ = pointer->us_msg_->GetExtensionAngle();
					pointer->FPS_ = pointer->us_msg_->GetFPS();
					pointer->FocusDepth_ = pointer->us_msg_->GetFocusDepth();
					pointer->FocusSpacing_ = pointer->us_msg_->GetFocusSpacing();
					pointer->FocusCount_ = (int) pointer->us_msg_->GetFocus_Count();
					pointer->ImageSize_ = pointer->us_msg_->GetImageSize();
					pointer->LineDensity_ = pointer->us_msg_->GetLineDensity();
					pointer->NumComponents_ = pointer->us_msg_->GetNumComponents();
					pointer->ProbeAngle_ = pointer->us_msg_->GetProbeAngle();
					pointer->ProbeID_ = pointer->us_msg_->GetProbeID();
					pointer->SamplingFrequency_ =
							pointer->us_msg_->GetSamplingFrequency();
					pointer->TransmitFrequency_ =
							pointer->us_msg_->GetTransmitFrequency();
					pointer->Pitch_ = pointer->us_msg_->GetPitch();
					pointer->Radius_ = pointer->us_msg_->GetRadius();
					pointer->ReferenceCount_ = pointer->us_msg_->GetReferenceCount();
					pointer->SteeringAngle_ = pointer->us_msg_->GetSteeringAngle();
					pointer->USDataType_ = pointer->us_msg_->GetUSDataType();
#ifdef DEBUG_OUTPUT
					std::cerr << "Device Name           : "
							<< pointer->us_msg_->GetDeviceName() << std::endl;
					std::cerr << "Scalar Type           : " << pointer->scalarType_
							<< std::endl;
					std::cerr << "Dimensions            : (" << pointer->size_[0] << ", "
							<< pointer->size_[1] << ", " << pointer->size_[2] << ")" << std::endl;
					std::cerr << "Spacing               : (" << pointer->spacing_[0]
							<< ", " << pointer->spacing_[1] << ", " << pointer->spacing_[2] << ")"
							<< std::endl;
					std::cerr << "Sub-Volume dimensions : (" << pointer->svsize_[0]
							<< ", " << pointer->svsize_[1] << ", " << pointer->svsize_[2] << ")"
							<< std::endl;
					std::cerr << "Sub-Volume offset     : (" << pointer->svoffset_[0]
							<< ", " << pointer->svoffset_[1] << ", " << pointer->svoffset_[2] << ")"
							<< std::endl;
					std::cerr << "Angle                 : " << pointer->Angle_
							<< std::endl;
					std::cerr << "FPS                   : " << pointer->FPS_ << std::endl;
					std::cerr << "FocusDepth            : " << pointer->FocusDepth_
							<< std::endl;
					std::cerr << "FocusSpacing          : " << pointer->FocusSpacing_
							<< std::endl;
					std::cerr << "FocusCount            : " << pointer->FocusCount_
							<< std::endl;
					std::cerr << "ImageSize             : " << pointer->ImageSize_
							<< std::endl;
					std::cerr << "LineDensity           : " << pointer->LineDensity_
							<< std::endl;
					std::cerr << "NumComponents         : " << pointer->NumComponents_
							<< std::endl;
					std::cerr << "ProbeAngle            : " << pointer->ProbeAngle_
							<< std::endl;
					std::cerr << "ProbeID               : " << pointer->ProbeID_
							<< std::endl;
					std::cerr << "SamplingFrequency     : " << pointer->SamplingFrequency_
							<< std::endl;
					std::cerr << "TransmitFrequency     : " << pointer->TransmitFrequency_
							<< std::endl;
					std::cerr << "Pitch                 : " << pointer->Pitch_
							<< std::endl;
					std::cerr << "Radius                : " << pointer->Radius_
							<< std::endl;
					std::cerr << "ReferenceCount        : " << pointer->ReferenceCount_
							<< std::endl;
					std::cerr << "SteeringAngle         : " << pointer->SteeringAngle_
							<< std::endl;
					std::cerr << "USDataType            : " << pointer->USDataType_
							<< std::endl;
#endif
					if (pointer->scalarType_ == igtl::ImageMessage::TYPE_UINT8) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_8UC(pointer->size_[2]));
					} else if (pointer->scalarType_ == igtl::ImageMessage::TYPE_INT8) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_8SC(pointer->size_[2]));
					} else if (pointer->scalarType_ == igtl::ImageMessage::TYPE_INT16) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_16SC(pointer->size_[2]));
					} else if (pointer->scalarType_ == igtl::ImageMessage::TYPE_UINT16) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_16UC(pointer->size_[2]));
					} else if (pointer->scalarType_ == igtl::ImageMessage::TYPE_INT32) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_32SC(pointer->size_[2]));
					} else if (pointer->scalarType_ == igtl::ImageMessage::TYPE_FLOAT32) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_32FC(pointer->size_[2]));
					} else if (pointer->scalarType_ == igtl::ImageMessage::TYPE_FLOAT64) {
						pointer->us_img_ = cv::Mat::zeros(pointer->size_[1], pointer->size_[0],
								CV_64FC(pointer->size_[2]));
					} else {
						perror("No supported type found\n");
						return;
					}

					if (pointer->NumComponents_ == 1) {

					    // If received data is RF, we need to push it into elastography class for computation
	                    if (strcmp(pointer->header_msg_->GetDeviceName(), "MD_US_2D")
	                            == 0) {
	                        time_t now;
	                        time(&now);
	                        pointer->elastography_->PushRFData(pointer->us_msg_->GetScalarPointer(),
	                                pointer->us_msg_->GetImageSize(), pointer->size_[1], pointer->size_[0],
	                                pointer->spacing_[0], pointer->spacing_[1], pointer->spacing_[2],
	                                pointer->LineDensity_, pointer->SamplingFrequency_,
	                                pointer->TransmitFrequency_, pointer->FPS_, now);
#ifdef DEBUG_OUTPUT
	                        printf("Pushing RF Data complete\n");
#endif
	                    }

						memcpy((void *) pointer->us_img_.data,
								pointer->us_msg_->GetScalarPointer(),
								pointer->us_msg_->GetImageSize());
			            cv::imshow("Received image", pointer->us_img_);
			            int key_pressed = cv::waitKey(100);
			            // whether or not to start writing image in a continuous way
			            if(key_pressed == 'c') {
			                pointer->continue_write_image_ = !pointer->continue_write_image_;
                            std::cout << "Continue writing flag toggled: " << pointer->continue_write_image_
                                    << std::endl;
			            } else if(key_pressed == 's') {
			                std::cout << "Capture a single frame\n";
			                pointer->continue_write_image_ = false;
                            cv::imwrite("/home/xingtong/Pictures/US/SINGLE_CAPTURED_" +
                                    std::string(pointer->header_msg_->GetDeviceName()) +
                                    boost::lexical_cast<std::string>(pointer->us_image_count_)
                            + ".tiff", pointer->us_img_);
                            pointer->us_image_count_++;
			            }

			            if (pointer->continue_write_image_ == true) {
                            cv::imwrite("/home/xingtong/Pictures/US/CONTINUOUS_CAPTURED_" +
                                    std::string(pointer->header_msg_->GetDeviceName()) +
                                    boost::lexical_cast<std::string>(pointer->us_image_count_)
                            + ".tiff", pointer->us_img_);
                            pointer->us_image_count_++;
			            }
					} else {
					    perror("Multiple components not supported yet\n");
					}

					bufferevent_setwatermark(bev, EV_READ,
							pointer->us_msg_->GetPackSize(),
							pointer->us_msg_->GetPackSize());

				} else {
					perror("US Image body unpacking failed\n");
					return;
				}

			} else {
				pointer->receiving_state_ = IGTL_client::RECEIVING_BODY;
				bufferevent_setwatermark(bev, EV_READ,
						pointer->us_msg_->GetPackBodySize()
								- pointer->bytes_received_,
						pointer->us_msg_->GetPackBodySize()
								- pointer->bytes_received_);
			} //c & igtl::MessageHeader::UNPACK_BODY

		}

	}

}

void eventcb_global(struct bufferevent *bev, short event, void *ctx) {
    cv::destroyAllWindows();
	if (event & BEV_EVENT_CONNECTED) {
		printf("We have connected to server!\n");
		bufferevent_setwatermark(bev, EV_READ,
				pointer->header_msg_->GetPackSize(),
				pointer->header_msg_->GetPackSize());
		bufferevent_enable(bev, EV_READ | EV_WRITE);
	} else if (event & BEV_EVENT_ERROR) {
		/* An error occured while connecting. */
		perror("Error occured while connecting the server!");
		bufferevent_free(bev);
		exit(1);
	} else if (event & BEV_EVENT_TIMEOUT) {
		/* must be a timeout event handle, handle it */
		perror("Time out occured!");

		// Haven't tested these two lines
		bufferevent_free(bev);
		exit(1);
	} else {
		perror("Event callback invoked");
	}
}

void IGTL_client::chatterCallback(const std_msgs::String::ConstPtr& msg)
{
  ROS_INFO("I heard: [%s]", msg->data.c_str());
}

void IGTL_client::ros_run() {
	ros::NodeHandle n;

	ros::Publisher chatter_pub = n.advertise < std_msgs::String
			> ("chatter", 100);
	ros::Subscriber sub = n.subscribe("chatter", 100, &IGTL_client::chatterCallback, this);
	ros::Rate loop_rate(1);
#ifdef DEBUG_OUTPUT
	printf("ros thread: %lu \n", getThreadId());
#endif
	/**
	 * A count of how many messages we have sent. This is used to create
	 * a unique string for each message.
	 */
	int count = 0;
	while (ros::ok()) {
		/**
		 * This is a message object. You stuff it with data, and then publish it.
		 */
		std_msgs::String msg;
		std::stringstream ss;
		ss << "String Message " << count;
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
#ifdef DEBUG_OUTPUT
	std::cout << "ros thread ended.\n";
#endif
}

void IGTL_client::socket_run() {
#ifdef DEBUG_OUTPUT
	printf("run thread: %lu \n", getThreadId());
#endif
	struct event_base *base;
	struct bufferevent *bev;
	struct sockaddr_in sin;
	base = event_base_new();

	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr((const char*)us_ip_address_);
	//"10.162.34.81"
    //23978 port for 2D Image
    //23877 port for RFServer

	if(RF_or_BMODE_ == 0) {
	    std::cout << "Connecting to RF server..\n";
	    sin.sin_port = htons(23877);
	} else {
	    std::cout << "Connecting to BMODE server..\n";
	    sin.sin_port = htons(23978);
	}

	//DeviceName is MD_US_2D or MD_BMODE_2D
	bev = bufferevent_socket_new(base, -1, BEV_OPT_CLOSE_ON_FREE);
	bufferevent_setcb(bev, readcb_global, NULL, eventcb_global, NULL);
	/* Note that you only get a BEV_EVENT_CONNECTED event if you launch the connect()
	 * attempt using bufferevent_socket_connect(). If you call connect() on your own,
	 * the connection gets reported as a write.
	 */
	if (bufferevent_socket_connect(bev, (struct sockaddr *) &sin, sizeof(sin))
			< 0) {
		perror("Failed to connect");
		bufferevent_free(bev);
		return;
	}
	event_base_dispatch(base);
	bufferevent_free(bev);

#ifdef DEBUG_OUTPUT
	std::cout << "socket thread ended.\n";
#endif
	return;
}
