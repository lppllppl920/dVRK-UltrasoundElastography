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

#ifndef _CONCURRENT_QUEUE_
#define _CONCURRENT_QUEUE_

#include <stdio.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <deque>
#include <list>
#include <queue>

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio.hpp>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>

using namespace std;

/*
 Source: http://www.justsoftwaresolutions.co.uk/threading/implementing-a-thread-safe-queue-using-condition-variables.html
 */

template<typename Data>
class concurrent_queue {
private:
	std::queue<Data> the_queue;
	mutable boost::mutex the_mutex;
	boost::condition_variable the_condition_variable;
	int l_lookahead;
public:

	concurrent_queue() {
		l_lookahead = 2;
	}
	void push(Data const& data) {
		boost::mutex::scoped_lock lock(the_mutex);

		cout << "Current queue size" << the_queue.size() << endl;
		/*if (l_lookahead == the_queue.size ()) {
		 the_queue.pop();
		 }*/

		the_queue.push(data);

		lock.unlock();
		the_condition_variable.notify_one();
	}

	bool empty() const {
		boost::mutex::scoped_lock lock(the_mutex);
		return the_queue.empty();
	}

	bool try_pop(Data& popped_value) {
		boost::mutex::scoped_lock lock(the_mutex);
		if (the_queue.empty()) {
			return false;
		}

		popped_value = the_queue.front();
		the_queue.pop();
		return true;
	}

	void wait_and_pop(Data& popped_value) {
		boost::mutex::scoped_lock lock(the_mutex);
		while (the_queue.empty()) {
			the_condition_variable.wait(lock);
		}

		popped_value = the_queue.front();
		the_queue.pop();
	}

	void wait_and_pop(Data& first_popped_value, Data& second_popped_value,
			int lookahead) {
		boost::mutex::scoped_lock lock(the_mutex);

		l_lookahead = lookahead + 1;
		while (the_queue.empty() || the_queue.size() < (l_lookahead)) {
			the_condition_variable.wait(lock);
		}

		first_popped_value = the_queue.front();
		second_popped_value = the_queue.back();
		the_queue.pop();
	}

};

#endif
