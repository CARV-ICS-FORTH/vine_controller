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
#ifndef FORMATER_HEADER
#define FORMATER_HEADER
#include <ostream>
#include <vector>

template <class CLASS>
class Formater
{
    public:
        template< class ITEM >
            friend ostream & operator<<(ostream & os,Formater< ITEM > fmt);
        Formater(const char * pre,const char * aft,const char * fin,const vector< CLASS > & vec)
            : pre(pre), aft(aft), fin(fin), vec(vec) {}
    private:
        const char * pre;				// Before output
        const char * aft;				// Between items
        const char * fin;				// After output
        const vector< CLASS > & vec;	// Stuff to output
};

template<class CLASS>
ostream & operator<<(ostream & os,Formater< CLASS > fmt)
{
    const char * delim = "";
    os << fmt.pre;
    for(typename vector< CLASS >::const_iterator itr = fmt.vec.begin() ;
            itr != fmt.vec.end() ; itr++)
    {
        os << delim << *itr;
        delim = fmt.aft;
    }
    os << fmt.fin;
    return os;
}
#endif
