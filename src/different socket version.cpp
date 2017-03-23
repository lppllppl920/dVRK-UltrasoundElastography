//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//#include <netinet/in.h>
//#include <netinet/tcp.h>
//#include <event.h>
//#include <sys/types.h>
//#include <sys/socket.h>
//#include <errno.h>
//#include <fcntl.h>
//#include <unistd.h>
//#include <arpa/inet.h> //inet_addr
//#include <netdb.h>
////#include<pthread.h> //for threading , link with lpthread
//
////内部函数，只能被本文件中的函数调用
//static short ListenPort = 8080;
//static long ListenAddr = INADDR_ANY;//任意地址，值就是0
//static int   MaxConnections = 1024;
//
//static int ServerSocket;
////创建event
//static struct event ServerEvent;
//
////将一个socket设置成非阻塞模式
////不论什么平台编写网络程序，都应该使用NONBLOCK socket的方式。这样可以保证你的程序至少不会在recv/send/accept/connect这些操作上发生block从而将整个网络服务都停下来
//int SetNonblock(int fd)
//{
//    int flags;
//    //fcntl()用来操作文件描述符的一些特性
//    if ((flags = fcntl(fd, F_GETFL)) == -1) {
//        return -1;
//    }
//
//    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
//        return -1;
//    }
//    return 0;
//}
//
////这个函数当客户端的socket可读时由libevent调用
//void ServerRead(int fd, short ev, void *arg)
//{
//    struct client *client = (struct client *)arg;
//    u_char buf[8196];
//    int len, wlen;
//
//    //会把参数fd 所指的文件传送count个字节到buf指针所指的内存中
//    len = read(fd, buf, sizeof(buf));
//    if (len == 0) {
//        /* 客户端断开连接，在这里移除读事件并且释放客户数据结构 */
//        printf("disconnected\n");
//        close(fd);
//        event_del(&ServerEvent);
//        free(client);
//        return;
//    } else if (len < 0) {
//        /* 出现了其它的错误，在这里关闭socket，移除事件并且释放客户数据结构 */
//        printf("socket fail %s\n", strerror(errno));
//        close(fd);
//        event_del(&ServerEvent);
//        free(client);
//        return;
//    }
//    /*
//       为了简便，我们直接将数据写回到客户端。通常我们不能在非阻塞的应用程序中这么做，
//       我们应该将数据放到队列中，等待可写事件的时候再写回客户端。
//       如果使用多个终端进行socket连接会出现错误socket fail Bad file descriptor
//     */
//    wlen = write(fd, buf, len);
//    if (wlen < len) {
//        printf("not all data write back to client\n");
//    }
//}
//
///*
//   当有一个连接请求准备被接受时，这个函数将被libevent调用并传递给三个变量:
//   int fd:触发事件的文件描述符.
//   short event:触发事件的类型EV_TIMEOUT,EV_SIGNAL, EV_READ, or EV_WRITE.
//   void* :由arg参数指定的变量.
//*/
//void ServerAccept(int fd, short ev, void *arg)
//{
//    int cfd;
//    struct sockaddr_in addr;
//    socklen_t addrlen = sizeof(addr);
//    int yes = 1;
//    int retval;
//
//    //将从连接请求队列中获得连接信息，创建新的套接字，并返回该套接字的文件描述符。
//    //新创建的套接字用于服务器与客户机的通信，而原来的套接字仍然处于监听状态。
//    //该函数的第一个参数指定处于监听状态的流套接字
//    cfd = accept(fd, (struct sockaddr *)&addr, &addrlen);
//    if (cfd == -1) {
//        printf("accept(): can not accept client connection");
//        return;
//    }
//    if (SetNonblock(cfd) == -1) {
//        close(cfd);
//        return;
//    }
//
//    //设置与某个套接字关联的选项
//    //参数二 IPPROTO_TCP:TCP选项
//    //参数三 TCP_NODELAY 不使用Nagle算法 选择立即发送数据而不是等待产生更多的数据然后再一次发送
//    //       更多参数TCP_NODELAY 和 TCP_CORK
//    //参数四 新选项TCP_NODELAY的值
//    if (setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == -1) {
//        printf("setsockopt(): TCP_NODELAY %s\n", strerror(errno));
//        close(cfd);
//        return;
//    }
//
//    event_set(&ServerEvent, cfd, EV_READ | EV_PERSIST, ServerRead, NULL);
//    event_add(&ServerEvent, NULL);
//
//    printf("Accepted connection from %s\n", inet_ntoa(addr.sin_addr));
//}
//int NewSocket(void)
//{
//    struct sockaddr_in sa;
//
//    //socket函数来创建一个能够进行网络通信的套接字。
//    //第一个参数指定应用程序使用的通信协议的协议族，对于TCP/IP协议族，该参数置AF_INET;
//    //第二个参数指定要创建的套接字类型
//    //流套接字类型为SOCK_STREAM、数据报套接字类型为SOCK_DGRAM、原始套接字SOCK_RAW
//    //第三个参数指定应用程序所使用的通信协议。
//    ServerSocket = socket(AF_INET, SOCK_STREAM, 0);
//    if (ServerSocket == -1) {
//        printf("socket(): can not create server socket\n");
//        return -1;
//    }
//    if (SetNonblock(ServerSocket) == -1) {
//        return -1;
//    }
//
//    //清空内存数据
//    memset(&sa, 0, sizeof(sa));
//    sa.sin_family = AF_INET;
//    //htons将一个无符号短整型数值转换为网络字节序
//    sa.sin_port = htons(ListenPort);
//    //htonl将主机的无符号长整形数转换成网络字节顺序
//    sa.sin_addr.s_addr = htonl(ListenAddr);
//
//    //(struct sockaddr*)&sa将sa强制转换为sockaddr类型的指针
//    /*struct sockaddr
//        数据结构用做bind、connect、recvfrom、sendto等函数的参数，指明地址信息。
//        但一般编程中并不直接针对此数据结构操作，而是使用另一个与sockaddr等价的数据结构 struct sockaddr_in
//        sockaddr_in和sockaddr是并列的结构，指向sockaddr_in的结构体的指针也可以指向
//        sockadd的结构体，并代替它。也就是说，你可以使用sockaddr_in建立你所需要的信息,
//        在最后用进行类型转换就可以了
//    */
//    //bind函数用于将套接字绑定到一个已知的地址上
//    if (bind(ServerSocket, (struct sockaddr*)&sa, sizeof(sa)) == -1) {
//        close(ServerSocket);
//        printf("bind(): can not bind server socket");
//        return -1;
//    }
//
//    //执行listen 之后套接字进入被动模式
//    //MaxConnections 连接请求队列的最大长度,队列满了以后，将拒绝新的连接请求
//    if (listen(ServerSocket, MaxConnections) == -1) {
//        printf("listen(): can not listen server socket");
//        close(ServerSocket);
//        return -1;
//    }
//
//    /*
//       event_set的参数：
//       + 参数1:  为要创建的event
//       + 参数2:  file descriptor，创建纯计时器可以设置其为-1，即宏evtimer_set定义的那样
//       + 参数3:  设置event种类，常用的EV_READ, EV_WRITE, EV_PERSIST, EV_SIGNAL, EV_TIMEOUT，纯计时器设置该参数为0
//       + 参数4:  event被激活之后触发的callback函数
//       + 参数5:  传递给callback函数的参数
//       备注：
//            如果初始化event的时候设置其为persistent的(设置了EV_PERSIST)，
//            则使用event_add将其添加到侦听事件集合后(pending状态)，
//            该event会持续保持pending状态，即该event可以无限次参加libevent的事件侦听。
//            每当其被激活触发callback函数执行之后，该event自动从active转回为pending状态，
//            继续参加libevent的侦听(当激活条件满足，又可以继续执行其callback)。
//            除非在代码中使用event_del()函数将该event从libevent的侦听事件集合中删除。
//            如果不通过设置EV_PERSIST使得event是persistent的，需要在event的callback中再次调用event_add
//            (即在每次pending变为active之后，在callback中再将其设置为pending)
//     */
//    event_set(&ServerEvent, ServerSocket, EV_READ | EV_PERSIST, ServerAccept, NULL);
//    //将event添加到libevent侦听的事件集中
//    if (event_add(&ServerEvent, 0) == -1) {
//        printf("event_add(): can not add accept event into libevent");
//        close(ServerSocket);
//        return -1;
//    }
//    return 0;
//}
//
//int main(int argc, char *argv[])
//{
//    int retval;
//
//    //初始化event base 使用默认的全局current_base
//    event_init();
//
//    retval = NewSocket();
//    if (retval == -1) {
//        exit(-1);
//    }
//    //event_dispatch() 启动事件队列系统，开始监听（并接受）请求
//    event_dispatch();
//
//    return 0;
//}

//#include<stdio.h>
//#include<string.h>    //strlen
//#include<stdlib.h>
//#include <unistd.h>
//#include<sys/socket.h>
//#include<arpa/inet.h> //inet_addr
//#include<netdb.h>
//#include<pthread.h> //for threading , link with lpthread
//
//void *connection_handler(void *socket_desc);
//
//int main(int argc , char *argv[])
//{
////----------------    Client   -------------------------
////    int socket_desc;
////    struct sockaddr_in server;
////    char *message;
////    char server_reply[2000];
////
////    //Create socket
////    socket_desc = socket(AF_INET , SOCK_STREAM, 0);
////    if (socket_desc == -1)
////    {
////        printf("Could not create socket");
////    }
////
////    server.sin_addr.s_addr = inet_addr("216.58.218.4");
////    server.sin_family = AF_INET;
////    server.sin_port = htons( 80 );
////
////    //Connect to remote server
////    if (connect(socket_desc , (struct sockaddr *)&server , sizeof(server)) < 0)
////    {
////        puts("connect error");
////        return 1;
////    }
////
////    puts("Connected\n");
////
////    //Send some data
////    message = "GET / HTTP/1.1\r\n\r\n";
////    if( send(socket_desc , message , strlen(message) , 0) < 0)
////    {
////        puts("Send failed");
////        return 1;
////    }
////    puts("Data Send\n");
////
////    //Receive a reply from the server
////    if( recv(socket_desc, server_reply , 2000 , 0) < 0)
////    {
////        puts("recv failed");
////    }
////
//////    if(read(socket_desc, server_reply , 2000) < 0)
//////    {
//////        puts("recv failed");
//////    }
////
////    puts("Reply received\n");
////    puts(server_reply);
////
////    close(socket_desc);
////
////
////    char *hostname = "www.google.com";
////    char ip[100];
////    struct hostent *he;
////    struct in_addr **addr_list;
////    int i;
////
////    if ( (he = gethostbyname( hostname ) ) == NULL)
////    {
////        //gethostbyname failed
////        herror("gethostbyname");
////        return 1;
////    }
////
////    //Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
////    addr_list = (struct in_addr **) he->h_addr_list;
////
////    for(i = 0; addr_list[i] != NULL; i++)
////    {
////        //Return the first one;
////        strcpy(ip , inet_ntoa(*addr_list[i]) );
////        printf("%s resolved to : %s" , hostname , ip);
////    }
////
////    return 0;
//
//// -------------- Server ----------------
//    int socket_desc , new_socket , c , *new_sock;
//    struct sockaddr_in server , client;
//    char *message;
//
//    //Create socket
//    socket_desc = socket(AF_INET , SOCK_STREAM , 0);
//    if (socket_desc == -1)
//    {
//        printf("Could not create socket");
//    }
//
//    //Prepare the sockaddr_in structure
//    server.sin_family = AF_INET;
//    server.sin_addr.s_addr = INADDR_ANY;
//    server.sin_port = htons( 8888 );
//
//    //Bind
//    if( bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0)
//    {
//        puts("bind failed");
//        return 1;
//    }
//    puts("bind done");
//
//    //Listen
//    listen(socket_desc , 3);
//
//    //Accept and incoming connection
//    puts("Waiting for incoming connections...");
//    c = sizeof(struct sockaddr_in);
//    while( (new_socket = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&c)) )
//    {
//        puts("Connection accepted");
//
//        //Reply to the client
//        message = (char*)"Hello Client , I have received your connection. And now I will assign a handler for you\n";
//        write(new_socket , message , strlen(message));
//
//        pthread_t sniffer_thread;
//        new_sock = (int*)malloc(sizeof(int) * 1);
//        *new_sock = new_socket;
//
//        if( pthread_create( &sniffer_thread , NULL ,  connection_handler , (void*) new_sock) < 0)
//        {
//            perror("could not create thread");
//            return 1;
//        }
//
//        //Now join the thread , so that we dont terminate before the thread
//        //pthread_join( sniffer_thread , NULL);
//        puts("Handler assigned");
//    }
//
//    if (new_socket<0)
//    {
//        perror("accept failed");
//        return 1;
//    }
//
//    return 0;
//}
//
///*
// * This will handle connection for each client
// * */
//void *connection_handler(void *socket_desc)
//{
//    //Get the socket descriptor
//    int sock = *(int*)socket_desc;
//    int read_size;
//    char *message , client_message[2000];
//
//    //Send some messages to the client
//    message = (char*) "Greetings! I am your connection handler\n";
//    write(sock , message , strlen(message));
//
//    message = (char*) "Now type something and i shall repeat what you type \n";
//    write(sock , message , strlen(message));
//
//    //Receive a message from client
//    while( (read_size = recv(sock , client_message , 2000 , 0)) > 0 )
//    {
//        //Send the message back to client
//        write(sock , client_message , strlen(client_message));
//    }
//
//    if(read_size == 0)
//    {
//        puts("Client disconnected");
//        fflush(stdout);
//    }
//    else if(read_size == -1)
//    {
//        perror("recv failed");
//    }
//
//    //Free the socket pointer
//    free(socket_desc);
//
//    return 0;
//}
