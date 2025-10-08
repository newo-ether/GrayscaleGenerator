#include"geopotential.h"
#include"mesh.h"
#include"threadpool.h"

using namespace geopotential;

constexpr double pi=3.14159265358979323;
constexpr size_t cn_tasksize=1024;

static void thread_work(void *param,size_t){
    auto *ptask=(std::function<void()>*)param;
    (*ptask)();
}

template<typename T,typename U>
void add_to(std::pair<T,U> &presult,const std::pair<T,U> &rhs){
    presult.first+=rhs.first;
    presult.second+=rhs.second;
}
template<typename T>
void add_to(T &presult,const T &rhs){
    presult+=rhs;
}

template<typename R>
class grouped_reducer:public std::function<void()>{
public:
    R local_sum;
    bool done{false};
    bool summed{false};
};

template<typename R,typename F,typename ...Args>
void integrate_parallel(ThreadPool::TaskGroup *pgroup,R *presult,const trigmesh &mesh,F func,const Args &...args){
    auto *pthreadpool=ThreadPool::get_thread_pool();
    if(!pthreadpool){
        memset(presult,0,sizeof(R));
        for(const auto &t:mesh)
            add_to(*presult,(t.*func)(args...));
        return;
    }
    size_t n_tasks=std::max(size_t(1),(mesh.size()+cn_tasksize/2)/cn_tasksize);
    typedef grouped_reducer<R> task_type;
    task_type *all_tasks=new task_type[n_tasks];
    std::mutex *local_mutex=new std::mutex;
    for(size_t i=0;i<n_tasks;++i){
        static_cast<std::function<void()>&>(all_tasks[i])=[&,local_mutex,all_tasks,i,n_tasks,func,presult](){
            size_t mesh_size=mesh.size();
            size_t imin=i*mesh_size/n_tasks;
            size_t imax=(i+1)*mesh_size/n_tasks;
            task_type &this_task=all_tasks[i];
            R &local_sum=this_task.local_sum;
            memset(&local_sum,0,sizeof(R));
            for(size_t j=imin;j<imax;++j)
                add_to(local_sum,(mesh[j].*func)(args...));
            bool free_all=false;
            local_mutex->lock();
            do{
                this_task.done=true;
                if(i+1<n_tasks){
                    task_type &next_task=all_tasks[i+1];
                    if(!next_task.summed)break;
                    add_to(local_sum,next_task.local_sum);
                }
                size_t ileft=i;
                while(ileft){
                    task_type &previous_task=all_tasks[ileft-1];
                    if(!previous_task.done)break;
                    add_to(previous_task.local_sum,all_tasks[ileft].local_sum);
                    --ileft;
                }
                all_tasks[ileft].summed=true;
                if(ileft==0){
                    *presult=all_tasks->local_sum;
                    free_all=true;
                }
            } while(0);
            local_mutex->unlock();
            if(free_all){
                delete local_mutex;
                delete[] all_tasks;
            }
        };
    }
    for(size_t i=0;i<n_tasks;++i)
        pthreadpool->add_task(thread_work,all_tasks+i,pgroup);
}

static void integrate_finish(ThreadPool::TaskGroup *pgroup){
    auto *pthreadpool=ThreadPool::get_thread_pool();
    if(pthreadpool)
        pthreadpool->wait_for_all(pgroup);
}

size_t geopotential::get_remaining_task(){
    auto *pthreadpool=ThreadPool::get_thread_pool();
    return pthreadpool?pthreadpool->busy():-1;
}

bool trigmesh::initialize(){
    for(int i=0;i<2;++i){
        ThreadPool::TaskGroup g;
        integrate_parallel(&g,&volume,*this,&trig::m);
        integrate_parallel(&g,&offset.x,*this,&trig::cx);
        integrate_parallel(&g,&offset.y,*this,&trig::cy);
        integrate_parallel(&g,&offset.z,*this,&trig::cz);
        integrate_finish(&g);
        if(!(volume>0&&volume!=INFINITY)){
            error_msg+="Non-finite-positive volume. Check model normals.\n";
            return false;
        }
        offset.x/=volume;
        offset.y/=volume;
        offset.z/=volume;
        if(i)break;
        for(auto &t:*this){
            t.p1-=offset;
            t.p2-=offset;
            t.p3-=offset;
        }
    }
    ref_radius=std::cbrt(volume*3/(4*pi));
    if(!(offset.norm()<1e-12*ref_radius)){
        error_msg+="Cannot eliminate barycentric offset. Is the model a closure?\n";
        return false;
    }
    double sa0=solid_angle(vec(0))/(4*pi);
    sa0-=std::round(sa0);
    if(!(std::abs(sa0)<1e-12)){
        error_msg+="Solid angle is not multiple of 4pi. Is the model a closure?\n";
        return false;
    }
    initialized=true;
    return true;
}

static std::string double_string(double x){
    char buf[32];
    sprintf(buf,"%.17e",x);
    return buf;
}
static std::string integer_string(int n){
    char buf[32];
    sprintf(buf,"%d",n);
    return buf;
}

//the normalization convention used by principia is differed from us...
static double principia_factor(int n,int m){
    double factor=1/double(2*n+1);
    for(int k=n-m;k<n+m;)factor*=++k;
    return m==0?-std::sqrt(factor):std::sqrt(factor/2);
}

bool Mesh::makeGeopotential(std::string &result,int max_degree,size_t n_thread) const{
    if(max_degree<2||6<max_degree){
        result="MaxDegree should be in [2,6]";
        return false;
    }

    class ThreadPoolGuard{
    public:
        ThreadPoolGuard(size_t n){
            if(n==1)return;
            ThreadPool::thread_local_pool_alloc();
            if(n)ThreadPool::get_thread_pool()->resize(n);
        }
        ~ThreadPoolGuard(){ ThreadPool::thread_local_pool_free(); }
    } _(n_thread);

    trigmesh gpmesh;
    for(const auto &t:triangles)
        gpmesh.emplace_back(
            vec(t.p[0].x,t.p[0].y,t.p[0].z),
            vec(t.p[1].x,t.p[1].y,t.p[1].z),
            vec(t.p[2].x,t.p[2].y,t.p[2].z));

    if(!gpmesh.initialize()){
        result=gpmesh.error_msg;
        return false;
    }

    auto gpdata=gpmesh.integrate_geopotential(max_degree);

    result="    reference_radius        = ";
    result+=double_string(gpmesh.ref_radius);
    result+=" km\n";
    for(int n=2;n<=max_degree;++n){
        auto data=gpdata[n];
        if(data.size()!=n*2+1){
            result="Degree not implemented";
            return false;
        }
        result+="    geopotential_row {\n";
        result+="      degree = ";
        result+=integer_string(n);
        result+="\n";
        for(int m=0;m<=n;++m){
            double pfactor=principia_factor(n,m);
            result+="      geopotential_column {\n";
            result+="        order = ";
            result+=integer_string(m);
            result+="\n";
            result+="        cos   = ";
            result+=double_string(pfactor*data[m]);
            result+="\n";
            result+="        sin   = ";
            result+=double_string(pfactor*(m==0?0:data[m+n]));
            result+="\n";
            result+="      }\n";
        }
        result+="    }\n";
    }
    return true;
}

double trig::solid_angle(const vec &r) const{
    vec r1=p1-r;
    vec r2=p2-r;
    vec r3=p3-r;
    double r1n=r1.norm();
    double r2n=r2.norm();
    double r3n=r3.norm();
    double r123=r1%(r2*r3);
    double L=2*std::atan2(r123,r1n*r2n*r3n+r1n*(r2%r3)+r2n*(r3%r1)+r3n*(r1%r2));
    return L;
}

double trigmesh::solid_angle(const vec &r) const{
    double result;
    ThreadPool::TaskGroup g;
    integrate_parallel(&g,&result,*this,&trig::solid_angle,r);
    integrate_finish(&g);
    return result;
}

std::pair<double,vec> trigmesh::gravity(const vec &r) const{
    std::pair<double,vec> result;
    ThreadPool::TaskGroup g;
    integrate_parallel(&g,&result,*this,&trig::gravity,r);
    integrate_finish(&g);
    return result;
}

std::pair<double,vec> trig::gravity(const vec &r) const{
    std::pair<double,vec> result(0,0);
    vec e1=p2-p1;
    vec e2=p3-p2;
    vec e3=p1-p3;
    vec n=e1*e2;
    double n2=n.normsqr();
    if(n2>0){
        auto lnterm=[](const vec &r1,const double &r1n,
                       const vec &r2,const double &r2n,
                       const vec &e1,const vec &e2){
            double e12=e1%e1;
            double e1n=std::sqrt(e12);
            double e1_r1=e1%r1;
            double e1nr1=e1n*r1n;
            double e1_r2=e1%r2;
            double e1nr2=e1n*r2n;
            double lnu=e1_r1+e1nr1;
            double lnd=e1_r2+e1nr2;
            return e1n>0&&lnu>0&&lnd>0?
                std::log(lnu/lnd)*(e1%e2*e1_r1-r1%e2*e12)/e1n
                :0;
        };
        vec r1=p1-r;
        vec r2=p2-r;
        vec r3=p3-r;
        double r1n=r1.norm();
        double r2n=r2.norm();
        double r3n=r3.norm();
        double r123=r1%(r2*r3);
        double L=2*std::atan2(r123,r1n*r2n*r3n+r1n*(r2%r3)+r2n*(r3%r1)+r3n*(r1%r2));
        L*=r123;
        L+=lnterm(r1,r1n,r2,r2n,e1,e2);
        L+=lnterm(r2,r2n,r3,r3n,e2,e3);
        L+=lnterm(r3,r3n,r1,r1n,e3,e1);
        L/=n2;
        result.first+=L*r123/2;
        result.second+=L*n;
    }

    return result;
}

std::map<int,std::vector<double>> trigmesh::integrate_geopotential(int max_degree) const{
    std::map<int,std::vector<double>> result;
    if(initialized&&max_degree<=6){
        for(int d=2;d<=max_degree;++d)
            result[d].resize(2*d+1);
        ThreadPool::TaskGroup g;
        double *presult;
        switch(max_degree)
        {
        case 6:
            presult=result.at(6).data();
            integrate_parallel(&g,presult++,*this,&trig::j6);
            integrate_parallel(&g,presult++,*this,&trig::c61);
            integrate_parallel(&g,presult++,*this,&trig::c62);
            integrate_parallel(&g,presult++,*this,&trig::c63);
            integrate_parallel(&g,presult++,*this,&trig::c64);
            integrate_parallel(&g,presult++,*this,&trig::c65);
            integrate_parallel(&g,presult++,*this,&trig::c66);
            integrate_parallel(&g,presult++,*this,&trig::s61);
            integrate_parallel(&g,presult++,*this,&trig::s62);
            integrate_parallel(&g,presult++,*this,&trig::s63);
            integrate_parallel(&g,presult++,*this,&trig::s64);
            integrate_parallel(&g,presult++,*this,&trig::s65);
            integrate_parallel(&g,presult++,*this,&trig::s66);
        case 5:
            presult=result.at(5).data();
            integrate_parallel(&g,presult++,*this,&trig::j5);
            integrate_parallel(&g,presult++,*this,&trig::c51);
            integrate_parallel(&g,presult++,*this,&trig::c52);
            integrate_parallel(&g,presult++,*this,&trig::c53);
            integrate_parallel(&g,presult++,*this,&trig::c54);
            integrate_parallel(&g,presult++,*this,&trig::c55);
            integrate_parallel(&g,presult++,*this,&trig::s51);
            integrate_parallel(&g,presult++,*this,&trig::s52);
            integrate_parallel(&g,presult++,*this,&trig::s53);
            integrate_parallel(&g,presult++,*this,&trig::s54);
            integrate_parallel(&g,presult++,*this,&trig::s55);
        case 4:
            presult=result.at(4).data();
            integrate_parallel(&g,presult++,*this,&trig::j4);
            integrate_parallel(&g,presult++,*this,&trig::c41);
            integrate_parallel(&g,presult++,*this,&trig::c42);
            integrate_parallel(&g,presult++,*this,&trig::c43);
            integrate_parallel(&g,presult++,*this,&trig::c44);
            integrate_parallel(&g,presult++,*this,&trig::s41);
            integrate_parallel(&g,presult++,*this,&trig::s42);
            integrate_parallel(&g,presult++,*this,&trig::s43);
            integrate_parallel(&g,presult++,*this,&trig::s44);
        case 3:
            presult=result.at(3).data();
            integrate_parallel(&g,presult++,*this,&trig::j3);
            integrate_parallel(&g,presult++,*this,&trig::c31);
            integrate_parallel(&g,presult++,*this,&trig::c32);
            integrate_parallel(&g,presult++,*this,&trig::c33);
            integrate_parallel(&g,presult++,*this,&trig::s31);
            integrate_parallel(&g,presult++,*this,&trig::s32);
            integrate_parallel(&g,presult++,*this,&trig::s33);
        case 2:
            presult=result.at(2).data();
            integrate_parallel(&g,presult++,*this,&trig::j2);
            integrate_parallel(&g,presult++,*this,&trig::c21);
            integrate_parallel(&g,presult++,*this,&trig::c22);
            integrate_parallel(&g,presult++,*this,&trig::s21);
            integrate_parallel(&g,presult++,*this,&trig::s22);
        default:
            break;
        }
        integrate_finish(&g);
        for(int d=2;d<=max_degree;++d){
            double multiplier=std::pow(ref_radius,double(-d))/volume;
            for(auto &c:result.at(d))
                c*=multiplier;
        }
    }
    return result;
}
