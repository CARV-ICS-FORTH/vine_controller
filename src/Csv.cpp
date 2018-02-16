/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1]
 *
 * Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 *  implied.
 * See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * Links:
 *  ------
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1]
*/
#include "Csv.h"

Csv :: Csv(const char * fname)
: start(std::chrono::system_clock::now()), ofs(fname)
{

}

Csv &  Csv :: print()
{
	std::chrono::duration<unsigned long long,std::nano> tstamp = std::chrono::system_clock::now()-start;
	ofs << tstamp.count();
	return (*this);
}
