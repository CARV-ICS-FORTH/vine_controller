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
#include "Accel.h"
#include <stdexcept>

Accel :: Accel()
: cpumask(-1)
{

}

void Accel :: pin(std::bitset<64> & available)
{
	auto legit = available & cpumask;
	if(legit.none())
		std::runtime_error("No available cores!");
	for(size_t c = 0 ; c < legit.size() ; c++)
	{
		if(legit[c])
		{
			cpumask = std::bitset<64>();
			cpumask = 1<<c;
			core = c;
			available.reset(c);
			break;
		}
	}
}

std::ostream & operator<<(std::ostream & os,const Accel & accel)
{
	os << "accel " << accel.type << " " << accel.name << " " << accel.core << " AnyJob ";
	if(accel.type == "GPU")
		os << accel.id <<" # " << accel.com << " ";
	os << std::endl;
	return os;
}

