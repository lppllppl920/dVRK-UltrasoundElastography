/***************************************************************************
 Copyright (c) 2014
 MUSiiC Laboratory
 Nishikant Deshmukh nishikant@jhu.edu, Emad M Boctor eboctor@jhmi.edu
 Johns Hopkins University

 For commercial use/licensing, please contact the authors

 Please see license.txt for further information.

 ***************************************************************************/

#ifndef _CRTDBG_MAP_ALLOC
#define _CRTDBG_MAP_ALLOC
#endif
/// Receiving RF frame from shared Memory
//#include "C:\ncc-vc\reader.h"

#include "reader.h"
#include "TrUE_Corr.h"

//#include "C:\ncc-vc\dynamic_program.h"

#define Max_size 1028
char tempBuf[Max_size];
int nRxLength(0);
int nRxIndex(0);

extern "C" {
void cuda_malloc_host(void **ptr, size_t ptr_size);
void cuda_free_local(void *ptr, char *name);
}
extern concurrent_queue<data_frame_queue *> in_queue;
extern concurrent_queue<data_frame_queue *> out_queue;
extern FrameHeader fhr;

// TODO: Client callback function
int ReceiveMsg(int numOfRun = 0, int taskInfo = 0, void* ptr = NULL,
		igtl::MessageBase::Pointer data1 = NULL, void* data2 = NULL,
		void* data3 = NULL) {
	//std::cout << igtl::GetMUSiiCTCPClientMessage(numOfRun, taskInfo, ptr, data1, data2, data3) << std::endl;
	if (taskInfo != M_TCP_DATA_RECEIVED)
		return 0;

	if (data1.IsNull())
		return 0;
	igtl::USMessage::Pointer ImgMsg = igtl::RetrieveMessage<
			igtl::USMessage::Pointer, igtl::USMessage>(data1);
	if (ImgMsg.IsNull())
		return 0;
	data_frame_queue *new_data; //put the newly arriving data into the out queue.
	data_frame_queue *old_data; //put the old data arriving into the queue.

	/*fhr.ss = 0.75 * 1000.0;
	 fhr.uly = 3 * 1000.0;
	 fhr.ulx = 1 * 1000.0;
	 fhr.ury = 2 * 1000.0;
	 fhr.urx = 0.0 * 1000.0;
	 fhr.brx = 0.035 * 10000.0;
	 fhr.bry = 0.035 * 10000.0;
	 fhr.txf = 5 * 1e6;
	 fhr.sf = 40 * 1e6;*/

	int size[3];          //image dimension
	float spacing[3];     //spacing (mm/pixel)
	int svsize[3];        //sub-volume size
	int svoffset[3];     //sub-volume offset
	int scalarType;       //scalar type

	scalarType = ImgMsg->GetScalarType();
	ImgMsg->GetDimensions(size); //pixel
	ImgMsg->GetSpacing(spacing); //spacing[0]:Width , spacing[1]: Depth, spacing[2]:Thickness mm
	ImgMsg->GetSubVolume(svsize, svoffset);

#ifdef PRINT_DEBUG
	std::cerr <<"===================================================================="<<std::endl;
	std::cerr << "Device Name           : " << ImgMsg->GetDeviceName() << std::endl;
	std::cerr << "Scalar Type           : " << scalarType << std::endl;
	std::cerr << "Dimensions            : ("
	<< size[0] << ", " << size[1] << ", " << size[2] << ")" << std::endl;
	std::cerr << "Spacing               : ("
	<< spacing[0] << ", " << spacing[1] << ", " << spacing[2] << ")" << std::endl;
	std::cerr << "Sub-Volume dimensions : ("
	<< svsize[0] << ", " << svsize[1] << ", " << svsize[2] << ")" << std::endl;
	std::cerr << "Sub-Volume offset     : ("
	<< svoffset[0] << ", " << svoffset[1] << ", " << svoffset[2] << ")" << std::endl;
	std::cerr <<"===================================================================="<<std::endl;
#endif

	ImgMsg->GetScalarPointer();

	char *tempdata;

	/*
	 * Allocate cuda pin memory to tempdata
	 */

	//TODO: change this memory allocation to malloc
	cuda_malloc_host((void **) &tempdata, ImgMsg->GetImageSize());
	//TODO: I think here you should use cudamemcpy?
	memcpy(tempdata, ImgMsg->GetScalarPointer(), ImgMsg->GetImageSize());

	new_data = new data_frame_queue;
	new_data->data = tempdata;
	new_data->height = size[1];
	new_data->width = size[0];
	new_data->number_frames = size[2];
	igtl::TimeStamp::Pointer ts = igtl::TimeStamp::New();
	ImgMsg->GetTimeStamp(ts);
	new_data->itime = ts->GetTimeStamp();
	new_data->fhr = fhr;

	new_data->spacing[0] = spacing[0];
	new_data->spacing[1] = spacing[1];
	new_data->spacing[2] = spacing[2];

	new_data->ImgMsg = ImgMsg;

	//ImgMsg->GetPitch();
	//ImgMsg->GetElements();
	//ImgMsg->GetSeteringAngle();

	in_queue.push(new_data);

	while (out_queue.try_pop(old_data)) {
		cuda_free_local(old_data->data, (char*) "queue");
		old_data->data = 0;
		delete old_data;
	}
	return 0;
}

int read_directory1(const char * path_name) {
	DIR *dir;
	struct dirent *ent;
	int count = 0;
	int i = 0;
	//igtl::MessageBase::Pointer ImgMsg;
	igtl::USMessage::Pointer ImgMsg;
	char igtl_file_name[100];
	if ((dir = opendir(path_name)) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
			count++;
		}
		closedir(dir);
		dir = opendir(path_name);
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
			if (i < 2) {
				i++;
				continue;
			}
			sprintf(igtl_file_name, "%s\\%s", path_name, ent->d_name);
			printf("%s", igtl_file_name);
			igtl::MUSiiCIGTLMsgFileIO::Pointer pFileIO =
					igtl::MUSiiCIGTLMsgFileIO::New();
			igtl::MessageBase::Pointer TempMsg = pFileIO->ReadSingleFile(
					igtl_file_name);
			//   ImgMsg = igtl::RetrieveMessage<igtl::USMessage::Pointer, igtl::USMessage>(TempMsg);
			// TempMsg->Pack();
			//ImgMsg = igtl::RetrieveMessage<igtl::USMessage::Pointer, igtl::USMessage>(TempMsg, false);

			ReceiveMsg(0, M_TCP_DATA_RECEIVED, NULL, TempMsg, NULL, NULL);
			i++;
		}
		closedir(dir);
	} else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
	return 0;
}

