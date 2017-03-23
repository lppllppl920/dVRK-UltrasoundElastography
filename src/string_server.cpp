/*=========================================================================
 Program:   OpenIGTLink -- Example for Tracker Server Program
 Language:  C++
 Copyright (c) Insight Software Consortium. All rights reserved.
 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.
 =========================================================================*/

#include <iostream>
#include <math.h>
#include <cstdlib>

#include "igtl/igtlOSUtil.h"
#include "igtl/igtlStringMessage.h"
#include "igtl/igtlServerSocket.h"

#define N_STRINGS 5

const char * testString[N_STRINGS] = { "OpenIGTLink", "Network",
        "Communication", "Protocol", "Image Guided Therapy", };

int main() {
    int port = 8080;
    double fps = 1;
    int interval = (int) (1000.0 / fps);

    igtl::StringMessage::Pointer stringMsg;
    stringMsg = igtl::StringMessage::New();
    stringMsg->SetDeviceName("StringMessage");

    igtl::ServerSocket::Pointer serverSocket;
    serverSocket = igtl::ServerSocket::New();
    int r = serverSocket->CreateServer(port);

    if (r < 0) {
        std::cerr << "Cannot create a server socket." << std::endl;
        exit(0);
    }

    igtl::Socket::Pointer socket;

    while (1) {
        //------------------------------------------------------------
        // Waiting for Connection
        socket = serverSocket->WaitForConnection();

        if (socket.IsNotNull()) // if client connected
        {
            //------------------------------------------------------------
            // loop
            for (int i = 0; i < 100; i++) {
                std::cout << "Sending string: " << testString[i % N_STRINGS]
                        << std::endl;
                stringMsg->SetString(testString[i % N_STRINGS]);
                stringMsg->Pack();
                socket->Send(stringMsg->GetPackPointer(),
                        stringMsg->GetPackSize());
                igtl::Sleep(interval); // wait
            }
        }
    }

    //------------------------------------------------------------
    // Close connection (The example code never reachs to this section ...)
    socket->CloseSocket();

}
