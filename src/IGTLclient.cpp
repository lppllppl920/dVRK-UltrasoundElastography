/*
 * IGTLclient.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: xingtong
 */

#include "IGTLclient.h"

IGTL_client::IGTL_client() {
	// Set this flag to true if you are ready to receive the header of the next image
	receiving_state_ = SEARCHING_HEADER;
	bytes_received_ = 0;

	// Create a message buffer to receive header
	header_msg_ = igtl::MessageHeader::New();
	us_msg_ = igtl::USMessage::New();
	us_image_count_ = 0;
}

IGTL_client::~IGTL_client() {

	cv::destroyAllWindows();
	socket_thread_->join();
	ros_thread_->join();
	ros::shutdown();
}

void IGTL_client::elastographyGenerating() {

}

void IGTL_client::run() {
	socket_thread_ = boost::make_shared < boost::thread
			> (boost::bind(&IGTL_client::socket_run, this));
	ros_thread_ = boost::make_shared < boost::thread
			> (boost::bind(&IGTL_client::ros_run, this));
}

unsigned long IGTL_client::getThreadId() {
	std::string threadId = boost::lexical_cast < std::string
			> (boost::this_thread::get_id());
	unsigned long threadNumber = 0;
	sscanf(threadId.c_str(), "%lx", &threadNumber);
	return threadNumber;
}

void readcb_global(struct bufferevent *bev, void *ctx) {
	printf("readcb thread: %lu \n", pointer->getThreadId());
	struct evbuffer *input, *output;
	char *dummy;
	char Device_type[DEVICE_TYPE_BYTES];
	unsigned char temp[SANITY_CHECK_BYTES];
	input = bufferevent_get_input(bev);
	output = bufferevent_get_output(bev);

	//for speeding up, we can first check the first two bytes and following 12 bytes to make sure the version number and type is right
	// which means possible version number, then go unpack and see if we get a right header
	if (pointer->receiving_state_ == IGTL_client::SEARCHING_HEADER) {

	    std::cout << "Searching header..\n";
		// Initialize candidate header
		if (pointer->bytes_received_ == 0) {
			candidate_header = (unsigned char *) malloc(SANITY_CHECK_BYTES);
			pointer->header_msg_->InitPack();
		}

		assert(SANITY_CHECK_BYTES > pointer->bytes_received_);

		int r = evbuffer_remove(input,
				(void *) (candidate_header + pointer->bytes_received_),
				(SANITY_CHECK_BYTES - pointer->bytes_received_));
		if (r > 0) {

			pointer->bytes_received_ += r;
			if (pointer->bytes_received_ >= SANITY_CHECK_BYTES) {
				// Let's do the sanity check
				unsigned short V = (unsigned short) (candidate_header[0] << 8)
						| (unsigned short) candidate_header[1];
				memcpy(Device_type, (void *) (candidate_header + 2),
						DEVICE_TYPE_BYTES);
				if ((V == 1 || V == 2) && (strcmp(Device_type, "IMAGE") == 0)) {
					std::cout << "Valid header found! \n";
					pointer->receiving_state_ = IGTL_client::RECEIVING_HEADER;
					memcpy(pointer->header_msg_->GetPackPointer(),
							(void *) candidate_header, SANITY_CHECK_BYTES);
					pointer->bytes_received_ = SANITY_CHECK_BYTES;

				} else {

					// Search the entire input buffer to find the valid header
					while (true) {
						// Invalid header, need to search for another one
						memcpy((void *) temp, (void *) (candidate_header + 1),
								(SANITY_CHECK_BYTES - 1));
						r = evbuffer_remove(input,
								(void *) (temp + SANITY_CHECK_BYTES - 1), 1);
						if (r <= 0) {
							memcpy(candidate_header, temp,
									SANITY_CHECK_BYTES - 1);
							pointer->receiving_state_ =
									IGTL_client::SEARCHING_HEADER;
							pointer->bytes_received_ = SANITY_CHECK_BYTES - 1;
							return;

						} else {
							memcpy(candidate_header, temp, SANITY_CHECK_BYTES);
							V = (unsigned short) (candidate_header[0] << 8)
									| (unsigned short) candidate_header[1];
							memcpy(Device_type, (void *) (candidate_header + 2),
									DEVICE_TYPE_BYTES);
							if ((V == 1 || V == 2)
									&& (strcmp(Device_type, "IMAGE") == 0)) {
								std::cout << "Valid header found! \n";
								pointer->receiving_state_ =
										IGTL_client::RECEIVING_HEADER;
								memcpy(pointer->header_msg_->GetPackPointer(),
										(void *) candidate_header,
										SANITY_CHECK_BYTES);
								pointer->bytes_received_ = SANITY_CHECK_BYTES;
								break;
							}
						}
					}

				}

			} else {
				// Not enought bytes, continue for more data
				pointer->receiving_state_ = IGTL_client::SEARCHING_HEADER;
				return;
			}

		} else {
			std::cerr
					<< "zero byte received, waiting to receive more data to complete a header sanity check\n";
			return;
		}

	}

	// Receiving header
	if (pointer->receiving_state_ == IGTL_client::RECEIVING_HEADER) {
	    std::cout << "Receiving header..\n";

		// If we start to receive a new header, first initialize the HeaderMessage object
		if (pointer->bytes_received_ == 0) {
			pointer->header_msg_->InitPack();
		}

		assert(pointer->header_msg_->GetPackSize() > pointer->bytes_received_);

		// Receive generic header from the socket
		int r =
				evbuffer_remove(input,
						(void*) ((unsigned char *) pointer->header_msg_->GetPackPointer()
								+ pointer->bytes_received_),
						pointer->header_msg_->GetPackSize()
								- pointer->bytes_received_);

		if (r <= 0) {
			std::cerr
					<< "zero byte received, waiting to receive more data to complete a message header\n";
			return;

		} else {

			pointer->bytes_received_ += r;
			std::cout << "Have read header bytes: " << pointer->bytes_received_
					<< " Ideal header bytes: "
					<< pointer->header_msg_->GetPackSize() << std::endl;

			// Header receiving complete
			if (pointer->bytes_received_
					>= pointer->header_msg_->GetPackSize()) {

				// Deserialize the header
				int c = pointer->header_msg_->Unpack(1);
				// If the header is
				if (c & igtl::MessageHeader::UNPACK_HEADER) {
					std::cout << "Device Type: "
							<< pointer->header_msg_->GetDeviceType()
							<< " Device Name: "
							<< pointer->header_msg_->GetDeviceName()
							<< std::endl;

					// US Message received
					if (strcmp(pointer->header_msg_->GetDeviceName(), "MD_US_2D")
							== 0) {
						pointer->receiving_state_ = IGTL_client::RECEIVING_BODY;
						pointer->bytes_received_ = 0;
					} else {
						// Didn't receive the desired type of message, skip it and research for desired header
					    std::cout << "Device Name is not MD_US_2D\n";
						pointer->receiving_state_ =
								IGTL_client::SEARCHING_HEADER;
						evbuffer_remove(input, dummy, MAX_LINE);
						pointer->bytes_received_ = 0;
						return;
					}
				} else {
					// If the header unpacking failed, which means we didn't find a valid header
					// We need to search for header again
					//TODO:
					pointer->receiving_state_ = IGTL_client::SEARCHING_HEADER;
					evbuffer_remove(input, dummy, MAX_LINE);
					pointer->bytes_received_ = 0;
					return;
				}

			} else {
				// Haven't received enought header data
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

	    std::cout << "Receiving Body..\n";
		// Initialize US Message when starting to receive body part
		if (pointer->bytes_received_ == 0) {
			pointer->us_msg_->SetMessageHeader(pointer->header_msg_);
			pointer->us_msg_->AllocatePack();
		}

		assert(pointer->us_msg_->GetPackBodySize() > pointer->bytes_received_);

		int r =
				evbuffer_remove(input,
						(void *) ((unsigned char*) pointer->us_msg_->GetPackBodyPointer()
								+ pointer->bytes_received_),
						pointer->us_msg_->GetPackBodySize()
								- pointer->bytes_received_);

		if (r <= 0) {
			std::cerr
					<< "zero byte received, waiting to receive more data to complete the US body part\n";
			return;
		} else {

			pointer->bytes_received_ += r;
			std::cout << "Have read image bytes: " << pointer->bytes_received_
					<< " Ideal image bytes: "
					<< pointer->us_msg_->GetPackBodySize() << std::endl;

			if (pointer->bytes_received_
					>= pointer->us_msg_->GetPackBodySize()) {
				// A complete US image received
				pointer->receiving_state_ = IGTL_client::RECEIVING_HEADER;
				pointer->bytes_received_ = 0;
				int c = pointer->us_msg_->Unpack(1);

				if (c & igtl::MessageHeader::UNPACK_BODY) {

					cv::namedWindow("Received image", CV_WINDOW_NORMAL);
					int size[3];          // image dimension
					float spacing[3];       // spacing (mm/pixel)
					int svsize[3];        // sub-volume size
					int svoffset[3];      // sub-volume offset
					int scalarType;       // scalar type

					scalarType = pointer->us_msg_->GetScalarType();
					pointer->us_msg_->GetDimensions(size);
					pointer->us_msg_->GetSpacing(spacing);
					pointer->us_msg_->GetSubVolume(svsize, svoffset);
					int Angle = pointer->us_msg_->GetExtensionAngle();
					int FPS = pointer->us_msg_->GetFPS();
					int FocusDepth = pointer->us_msg_->GetFocusDepth();
					int FocusSpacing = pointer->us_msg_->GetFocusSpacing();
					int FocusCount = (int) pointer->us_msg_->GetFocus_Count();
					int ImageSize = pointer->us_msg_->GetImageSize();
					int LineDensity = pointer->us_msg_->GetLineDensity();
					int NumComponents = pointer->us_msg_->GetNumComponents();
					int ProbeAngle = pointer->us_msg_->GetProbeAngle();
					int ProbeID = pointer->us_msg_->GetProbeID();
					int SamplingFrequency =
							pointer->us_msg_->GetSamplingFrequency();
					int TransmitFrequency =
							pointer->us_msg_->GetTransmitFrequency();
					int Pitch = pointer->us_msg_->GetPitch();
					int Radius = pointer->us_msg_->GetRadius();
					int ReferenceCount = pointer->us_msg_->GetReferenceCount();
					int SteeringAngle = pointer->us_msg_->GetSteeringAngle();
					int USDataType = pointer->us_msg_->GetUSDataType();

					std::cerr << "Device Name           : "
							<< pointer->us_msg_->GetDeviceName() << std::endl;
					std::cerr << "Scalar Type           : " << scalarType
							<< std::endl;
					std::cerr << "Dimensions            : (" << size[0] << ", "
							<< size[1] << ", " << size[2] << ")" << std::endl;
					std::cerr << "Spacing               : (" << spacing[0]
							<< ", " << spacing[1] << ", " << spacing[2] << ")"
							<< std::endl;
					std::cerr << "Sub-Volume dimensions : (" << svsize[0]
							<< ", " << svsize[1] << ", " << svsize[2] << ")"
							<< std::endl;
					std::cerr << "Sub-Volume offset     : (" << svoffset[0]
							<< ", " << svoffset[1] << ", " << svoffset[2] << ")"
							<< std::endl;
					std::cerr << "Angle                 : " << Angle
							<< std::endl;
					std::cerr << "FPS                   : " << FPS << std::endl;
					std::cerr << "FocusDepth            : " << FocusDepth
							<< std::endl;
					std::cerr << "FocusSpacing          : " << FocusSpacing
							<< std::endl;
					std::cerr << "FocusCount            : " << FocusCount
							<< std::endl;
					std::cerr << "ImageSize             : " << ImageSize
							<< std::endl;
					std::cerr << "LineDensity           : " << LineDensity
							<< std::endl;
					std::cerr << "NumComponents         : " << NumComponents
							<< std::endl;
					std::cerr << "ProbeAngle            : " << ProbeAngle
							<< std::endl;
					std::cerr << "ProbeID               : " << ProbeID
							<< std::endl;
					std::cerr << "SamplingFrequency     : " << SamplingFrequency
							<< std::endl;
					std::cerr << "TransmitFrequency     : " << TransmitFrequency
							<< std::endl;
					std::cerr << "Pitch                 : " << Pitch
							<< std::endl;
					std::cerr << "Radius                : " << Radius
							<< std::endl;
					std::cerr << "ReferenceCount        : " << ReferenceCount
							<< std::endl;
					std::cerr << "SteeringAngle         : " << SteeringAngle
							<< std::endl;
					std::cerr << "USDataType            : " << USDataType
							<< std::endl;

					if (scalarType == igtl::ImageMessage::TYPE_UINT8) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_8UC(size[2]));
					} else if (scalarType == igtl::ImageMessage::TYPE_INT8) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_8SC(size[2]));
					} else if (scalarType == igtl::ImageMessage::TYPE_INT16) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_16SC(size[2]));
					} else if (scalarType == igtl::ImageMessage::TYPE_UINT16) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_16UC(size[2]));
					} else if (scalarType == igtl::ImageMessage::TYPE_INT32) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_32SC(size[2]));
					} else if (scalarType == igtl::ImageMessage::TYPE_FLOAT32) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_32FC(size[2]));
					} else if (scalarType == igtl::ImageMessage::TYPE_FLOAT64) {
						pointer->us_img_ = cv::Mat::zeros(size[0], size[1],
								CV_64FC(size[2]));
					} else {
						perror("No supported type found");
						return;
					}

					if (NumComponents == 1) {
						memcpy((void *) pointer->us_img_.data,
								pointer->us_msg_->GetScalarPointer(),
								pointer->us_msg_->GetImageSize());

			            cv::Mat Display_image, Display_image_normalized;
			            pointer->us_img_.convertTo(Display_image, CV_32F);

			            double min, max;
			            std::cout << "min: " << min << " max: " << max << std::endl;
			            cv::minMaxLoc(Display_image, &min, &max);
			            Display_image_normalized = (Display_image - min) / (max - min) * 255.0;
			            cv::imshow("Received image", Display_image);
			            cv::waitKey(100);
			            if(pointer->us_image_count_ < 100) {

			                cv::imwrite("/home/xingtong/Pictures/US/" +
			                        boost::lexical_cast<std::string>(pointer->us_image_count_)
			                + ".tiff",pointer->us_img_);
			            }
			            pointer->us_image_count_++;
					}

					bufferevent_setwatermark(bev, EV_READ,
							pointer->us_msg_->GetPackSize(),
							pointer->us_msg_->GetPackSize());

				} else {
					std::cerr << "US Image body unpacking failed.\n";
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
		/* We're connected to 127.0.0.1:8080.   Ordinarily we'd do
		 something here, like start reading or writing. */
		printf("We are connected to server!\n");
		//TODO: This needs to be set according to the image size
		// The minimum value is how many bytes we need to have before invoking the read callback function
		// For robustness consideration, suppose we don't know the exact number of bytes
		// we need to complete receiving an image for now
		bufferevent_setwatermark(bev, EV_READ,
				pointer->header_msg_->GetPackSize(),
				pointer->header_msg_->GetPackSize());
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
//TODO: If the header cannot be deserialized, we need to search for the valid header in the following bit stream
//void IGTL_client::readcb(struct bufferevent *bev, void *ctx) {
//
//	printf("readcb thread: %lu \n", getThreadId());
//	struct evbuffer *input, *output;
//	char *dummy;
//	input = bufferevent_get_input(bev);
//	output = bufferevent_get_output(bev);
//
//	// Receiving header
//	if (receiving_state_ == RECEIVING_HEADER) {
//
//		// If we start to receive a new header, first initialize the HeaderMessage object
//		if (bytes_received_ == 0) {
//			header_msg_->InitPack();
//		}
//
//		assert(header_msg_->GetPackSize() > bytes_received_);
//
//		// Receive generic header from the socket
//		int r = evbuffer_remove(input,
//				(void*) ((unsigned char *) header_msg_->GetPackPointer()
//						+ bytes_received_),
//				header_msg_->GetPackSize() - bytes_received_);
//
//		if (r <= 0) {
//			std::cerr
//					<< "zero byte received, waiting to receive more data to complete a message header\n";
//			return;
//
//		} else {
//
//			bytes_received_ += r;
//			std::cout << "Have read header bytes: " << bytes_received_
//					<< " Ideal header bytes: " << header_msg_->GetPackSize()
//					<< std::endl;
//
//			// Header receiving complete
//			if (bytes_received_ == header_msg_->GetPackSize()) {
//
//				// Deserialize the header
//				header_msg_->Unpack(1);
//				std::cout << "Device Type: " << header_msg_->GetDeviceType()
//						<< std::endl;
//
//				std::cout << "Receiving Device Name: " << header_msg_->GetDeviceName() << std::endl;
//				// US Message received
//				if (strcmp(header_msg_->GetDeviceName(), "M_US_2D") == 0) {
//					receiving_state_ = RECEIVING_BODY;
//					bytes_received_ = 0;
//				} else {
//					// Didn't receive the desired type of message
//					receiving_state_ = RECEIVING_HEADER;
//					evbuffer_remove(input, dummy, MAX_LINE);
//					bytes_received_ = 0;
//				}
//
//			} else {
//				// Haven't received enought header data
//				// Continue with it for the next readcb
//				receiving_state_ = RECEIVING_HEADER;
//				bufferevent_setwatermark(bev, EV_READ,
//						header_msg_->GetPackSize() - bytes_received_,
//						header_msg_->GetPackSize() - bytes_received_);
//				return;
//			}
//
//		}
//	}
//
//	// Receiving body
//	if (receiving_state_ == RECEIVING_BODY) {
//
//		// Initialize US Message when starting to receive body part
//		if (bytes_received_ == 0) {
//			us_msg_->SetMessageHeader(header_msg_);
//			us_msg_->AllocatePack();
//		}
//
//		assert(us_msg_->GetPackBodySize() > bytes_received_);
//
//		int r = evbuffer_remove(input,
//				(void *) ((unsigned char*) us_msg_->GetPackBodyPointer()
//						+ bytes_received_),
//				us_msg_->GetPackBodySize() - bytes_received_);
//
//		if (r <= 0) {
//			std::cerr
//					<< "zero byte received, waiting to receive more data to complete the US body part\n";
//			return;
//		} else {
//
//			bytes_received_ += r;
//			std::cout << "Have read image bytes: " << bytes_received_
//					<< " Ideal image bytes: " << us_msg_->GetPackBodySize()
//					<< std::endl;
//
//			if (us_msg_->GetPackBodySize() == bytes_received_) {
//				// A complete US image received
//				receiving_state_ = RECEIVING_HEADER;
//				bytes_received_ = 0;
//				int c = us_msg_->Unpack(1);
//
//				if (c & igtl::MessageHeader::UNPACK_BODY) {
//
//					cv::namedWindow("Received image", CV_WINDOW_NORMAL);
//					int size[3];          // image dimension
//					float spacing[3];       // spacing (mm/pixel)
//					int svsize[3];        // sub-volume size
//					int svoffset[3];      // sub-volume offset
//					int scalarType;       // scalar type
//
//					scalarType = us_msg_->GetScalarType();
//					us_msg_->GetDimensions(size);
//					us_msg_->GetSpacing(spacing);
//					us_msg_->GetSubVolume(svsize, svoffset);
//					int Angle = us_msg_->GetExtensionAngle();
//					int FPS = us_msg_->GetFPS();
//					int FocusDepth = us_msg_->GetFocusDepth();
//					int FocusSpacing = us_msg_->GetFocusSpacing();
//					int FocusCount = (int) us_msg_->GetFocus_Count();
//					int ImageSize = us_msg_->GetImageSize();
//					int LineDensity = us_msg_->GetLineDensity();
//					int NumComponents = us_msg_->GetNumComponents();
//					int ProbeAngle = us_msg_->GetProbeAngle();
//					int ProbeID = us_msg_->GetProbeID();
//					int SamplingFrequency = us_msg_->GetSamplingFrequency();
//					int TransmitFrequency = us_msg_->GetTransmitFrequency();
//					int Pitch = us_msg_->GetPitch();
//					int Radius = us_msg_->GetRadius();
//					int ReferenceCount = us_msg_->GetReferenceCount();
//					int SteeringAngle = us_msg_->GetSteeringAngle();
//					int USDataType = us_msg_->GetUSDataType();
//
//					std::cerr << "Device Name           : "
//							<< us_msg_->GetDeviceName() << std::endl;
//					std::cerr << "Scalar Type           : " << scalarType
//							<< std::endl;
//					std::cerr << "Dimensions            : (" << size[0] << ", "
//							<< size[1] << ", " << size[2] << ")" << std::endl;
//					std::cerr << "Spacing               : (" << spacing[0]
//							<< ", " << spacing[1] << ", " << spacing[2] << ")"
//							<< std::endl;
//					std::cerr << "Sub-Volume dimensions : (" << svsize[0]
//							<< ", " << svsize[1] << ", " << svsize[2] << ")"
//							<< std::endl;
//					std::cerr << "Sub-Volume offset     : (" << svoffset[0]
//							<< ", " << svoffset[1] << ", " << svoffset[2] << ")"
//							<< std::endl;
//					std::cerr << "Angle                 : " << Angle
//							<< std::endl;
//					std::cerr << "FPS                   : " << FPS << std::endl;
//					std::cerr << "FocusDepth            : " << FocusDepth
//							<< std::endl;
//					std::cerr << "FocusSpacing          : " << FocusSpacing
//							<< std::endl;
//					std::cerr << "FocusCount            : " << FocusCount
//							<< std::endl;
//					std::cerr << "ImageSize             : " << ImageSize
//							<< std::endl;
//					std::cerr << "LineDensity           : " << LineDensity
//							<< std::endl;
//					std::cerr << "NumComponents         : " << NumComponents
//							<< std::endl;
//					std::cerr << "ProbeAngle            : " << ProbeAngle
//							<< std::endl;
//					std::cerr << "ProbeID               : " << ProbeID
//							<< std::endl;
//					std::cerr << "SamplingFrequency     : " << SamplingFrequency
//							<< std::endl;
//					std::cerr << "TransmitFrequency     : " << TransmitFrequency
//							<< std::endl;
//					std::cerr << "Pitch                 : " << Pitch
//							<< std::endl;
//					std::cerr << "Radius                : " << Radius
//							<< std::endl;
//					std::cerr << "ReferenceCount        : " << ReferenceCount
//							<< std::endl;
//					std::cerr << "SteeringAngle         : " << SteeringAngle
//							<< std::endl;
//					std::cerr << "USDataType            : " << USDataType
//							<< std::endl;
//
//					if (scalarType == igtl::ImageMessage::TYPE_UINT8) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_8UC(size[2]));
//					} else if (scalarType == igtl::ImageMessage::TYPE_INT8) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_8SC(size[2]));
//					} else if (scalarType == igtl::ImageMessage::TYPE_INT16) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_16SC(size[2]));
//					} else if (scalarType == igtl::ImageMessage::TYPE_UINT16) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_16UC(size[2]));
//					} else if (scalarType == igtl::ImageMessage::TYPE_INT32) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_32SC(size[2]));
//					} else if (scalarType == igtl::ImageMessage::TYPE_FLOAT32) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_32FC(size[2]));
//					} else if (scalarType == igtl::ImageMessage::TYPE_FLOAT64) {
//						us_img_ = cv::Mat::zeros(size[0], size[1],
//								CV_64FC(size[2]));
//					} else {
//						perror("No supported type found");
//						return;
//					}
//
//					if (NumComponents == 1) {
//						memcpy((void *) us_img_.data,
//								us_msg_->GetScalarPointer(),
//								us_msg_->GetImageSize());
//						cv::imshow("Received image", us_img_);
//						cv::waitKey(100);
//					}
//
//					bufferevent_setwatermark(bev, EV_READ,
//							us_msg_->GetPackSize(), us_msg_->GetPackSize());
//
//				} else {
//					std::cerr << "US Image body unpacking failed.\n";
//					return;
//				}
//
//			} else {
//				receiving_state_ = RECEIVING_BODY;
//				bufferevent_setwatermark(bev, EV_READ,
//						us_msg_->GetPackBodySize() - bytes_received_,
//						us_msg_->GetPackBodySize() - bytes_received_);
//			} //c & igtl::MessageHeader::UNPACK_BODY
//
//		}
//
//	}
//
//	/* // Check data type and receive data body
//	 if (strcmp(headerMsg->GetDeviceType(), "TRANSFORM") == 0) {
//	 ReceiveTransform(input, headerMsg);
//	 } else if (strcmp(headerMsg->GetDeviceType(), "POSITION") == 0) {
//	 ReceivePosition(input, headerMsg);
//	 } else if (strcmp(headerMsg->GetDeviceType(), "IMAGE") == 0) {
//	 ReceiveImage(input, headerMsg);
//	 } else if (strcmp(headerMsg->GetDeviceType(), "STATUS") == 0) {
//	 ReceiveStatus(input, headerMsg);
//	 }
//	 #if OpenIGTLink_PROTOCOL_VERSION >= 2
//	 else if (strcmp(headerMsg->GetDeviceType(), "POINT") == 0) {
//	 ReceivePoint(input, headerMsg);
//	 } else if (strcmp(headerMsg->GetDeviceType(), "STRING") == 0) {
//	 ReceiveString(input, headerMsg);
//	 } else if (strcmp(headerMsg->GetDeviceType(), "BIND") == 0) {
//	 ReceiveBind(input, headerMsg);
//	 }
//	 #endif //OpenIGTLink_PROTOCOL_VERSION >= 2
//	 else if (strcmp(headerMsg->GetDeviceName(), "US") == 0) {
//	 ReceiveUS(input, headerMsg);
//	 } else {
//	 // if the data type is unknown, skip reading.
//	 std::cerr << "Receiving unknown device type: "
//	 << headerMsg->GetDeviceType() << std::endl;
//	 evbuffer_remove(input, dummy, MAX_LINE);
//	 } */
//
//}

//void IGTL_client::eventcb(struct bufferevent *bev, short event, void *ctx) {
//
//	cv::destroyAllWindows();
//	if (event & BEV_EVENT_CONNECTED) {
//		/* We're connected to 127.0.0.1:8080.   Ordinarily we'd do
//		 something here, like start reading or writing. */
//		printf("We are connected to server!\n");
//		//TODO: This needs to be set according to the image size
//		// The minimum value is how many bytes we need to have before invoking the read callback function
//		// For robustness consideration, suppose we don't know the exact number of bytes
//		// we need to complete receiving an image for now
//		bufferevent_setwatermark(bev, EV_READ, header_msg_->GetPackSize(),
//				header_msg_->GetPackSize());
//		bufferevent_enable(bev, EV_READ | EV_WRITE);
//	} else if (event & BEV_EVENT_ERROR) {
//		/* An error occured while connecting. */
//		perror("Error occured while connecting the server!");
//		bufferevent_free(bev);
//	} else if (event & BEV_EVENT_TIMEOUT) {
//		/* must be a timeout event handle, handle it */
//		perror("Time out occured!");
//	} else {
//		perror("Event callback invoked");
//	}
//
//}

void IGTL_client::ros_run() {
	ros::NodeHandle n;

	ros::Publisher chatter_pub = n.advertise < std_msgs::String
			> ("chatter", 1000);
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
	std::cout << "ros thread ended.\n";
}

void IGTL_client::socket_run() {
	printf("run thread: %lu \n", getThreadId());
	struct event_base *base;
	struct bufferevent *bev;
	struct sockaddr_in sin;
	base = event_base_new();

	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr("10.162.34.81"); //htonl(0x7f000001); /* 127.0.0.1 */
	sin.sin_port = htons(23877); /* Port 8080 */

	//23877 port for RFServer
	//DeviceName is M_US_2D

	bev = bufferevent_socket_new(base, -1, BEV_OPT_CLOSE_ON_FREE);
	//boost::function<void (struct bufferevent*, void*)> f_read( boost::bind(&readcb_test, this, _1, _2) );
	//boost::function<void (struct bufferevent*, short int, void*)> f_event( boost::bind(&eventcb_test, this, _1, _2, _3) );

	bufferevent_setcb(bev, readcb_global, NULL, eventcb_global, NULL);
//    bufferevent_setcb(bev, , NULL, IGTL_client::eventcb,
//            NULL);
	/* Note that you only get a BEV_EVENT_CONNECTED event if you launch the connect()
	 * attempt using bufferevent_socket_connect(). If you call connect() on your own,
	 * the connection gets reported as a write.
	 */
	if (bufferevent_socket_connect(bev, (struct sockaddr *) &sin, sizeof(sin))
			< 0) {
		/* Error starting connection */
		perror("Failed to connect");
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

			cv::Mat Display_image, Display_image_normalized;
			image.convertTo(Display_image, CV_32F);

			double min, max;
			std::cout << "min: " << min << " max: " << max << std::endl;
			cv::minMaxLoc(Display_image, &min, &max);
			Display_image_normalized = (Display_image - min) / (max - min) * 255.0;
			cv::imshow("Received image: ", Display_image_normalized);
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

	if (c & igtl::MessageHeader::UNPACK_BODY)// if CRC check is OK
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

	if (c & igtl::MessageHeader::UNPACK_BODY)// if CRC check is OK
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

	if (c & igtl::MessageHeader::UNPACK_BODY)// if CRC check is OK
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
