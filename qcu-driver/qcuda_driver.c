/*
   The driver help send cuda function parameters to device.
   If the parameter is number, do nothing and send out.
   Otherwise, if the parameter is pointer address of data,
   you should copy data from user space to kernel space 
   and replace pointer address with gpa.
 */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/err.h>
#include <linux/virtio.h>
#include <linux/virtio_ids.h>
#include <linux/virtio_pci.h>
#include <linux/scatterlist.h>
#include <linux/random.h>
#include <linux/io.h>
#include <linux/sort.h>

#include <linux/virtio_config.h>

#include "qcuda_common.h"

#if 0
#define pfunc() printk(KERN_WARNING "qcuda_driver info: %s : %d\n", __func__, __LINE__)
#else
#define pfunc()
#endif

#if 0
#define ptrace(fmt, arg...) \
    printk("    ### " fmt, ##arg)
#else
#define ptrace(fmt, arg...)
#endif


#define error(fmt, arg...) \
    printk(KERN_ERR "### func= %-30s ,line= %-4d ," fmt, __func__,  __LINE__, ##arg)

#ifndef MIN
#define MIN(a, b) (((a)<(b))? (a):(b))
#endif

struct virtio_qcuda {
    struct virtio_device *vdev;
    struct virtqueue *vq;
    spinlock_t lock;
};

struct virtio_qcuda *qcu;

//TODO: 
//could be a structure let different processes open the same file
typedef int virtio_qc_file;

struct virtio_qc_page {
    unsigned int numOfblocks;
    unsigned long uvm_start;
    unsigned long uvm_end;
    unsigned long *page;
    struct list_head list;
    virtio_qc_file file;
    uint64_t data;
};

struct virtio_qc_mmap {
//	int 					block_id;
    unsigned int block_size;
    struct list_head head;
    unsigned int order;
    struct virtio_qc_page *group;
};



////////////////////////////////////////////////////////////////////////////////
///	General Function
////////////////////////////////////////////////////////////////////////////////

static inline unsigned long copy_from_user_safe(void *to, const void __user

*from,
unsigned long n
){
unsigned long err;

if( from==NULL || n==0 ){
memset(to,
0, n);
return 0;
}

err = copy_from_user(to, from, n);
if( err ){
error("copy_from_user is could not copy  %lu bytes\n", err);
BUG_ON(1);
}

return
err;
}

static inline unsigned long copy_to_user_safe(void __user

*to,
const void *from,
unsigned long n
) {
unsigned long err;
if( to==NULL || n==0 )
return 0;

err = copy_to_user(to, from, n);
if( err ){
error("copy_to_user is could not copy  %lu bytes\n", err);
}
return
err;
}

static inline void *kzalloc_safe(size_t size) {
    void *ret;

    ret = kzalloc(size, GFP_KERNEL);
    if (!ret) {
        error("kzalloc failed, size= %lu\n", size);
        BUG_ON(1);
    }

    return ret;
}

static inline void *kmalloc_safe(size_t size) {
    void *ret;

    ret = kmalloc(size, GFP_KERNEL);
    if (!ret) {
        error("kmalloc failed, size= %lu\n", size);
        BUG_ON(1);
    }

    return ret;
}

static void gpa_to_user(void *user, uint64_t gpa, uint32_t size) {
    uint32_t i, len;
    uint64_t *gpa_array;
    void *gva;

    if (size > QCU_KMALLOC_MAX_SIZE) {
        gpa_array = (uint64_t *) phys_to_virt((phys_addr_t) gpa);
        for (i = 0; size > 0; i++) {
            len = MIN(size, QCU_KMALLOC_MAX_SIZE);

            gva = phys_to_virt(gpa_array[i]);
            copy_to_user_safe(user, gva, len);

            user += len;
            size -= len;
        }
    } else {
        gva = phys_to_virt((phys_addr_t) gpa);
        copy_to_user_safe(user, gva, size);
    }
}

static uint64_t user_to_gpa_small(uint64_t from, uint32_t n) {
    void *gva;

    ptrace("from= %p, size= %u\n", (void *) from, n);

    gva = kmalloc_safe(n);

    if (from) { // there is data needed to copy
        copy_from_user_safe(gva, (const void *) from, n);
    }

    return (uint64_t) virt_to_phys(gva);
}

static uint64_t user_to_gpa_large(uint64_t from, uint32_t size) {
    uint32_t i, order, len;
    uint64_t *gpa_array;
    void *gva;

    ptrace("from= %p, size= %u\n", (void *) from, size);

    order = (size >> QCU_KMALLOC_SHIFT_BIT) +
            ((size & (QCU_KMALLOC_MAX_SIZE - 1)) > 0);

    gpa_array = (uint64_t *) kmalloc_safe(sizeof(uint64_t) * order);

    for (i = 0; size > 0; i++) {
        len = MIN(size, QCU_KMALLOC_MAX_SIZE);
        gva = kmalloc_safe(len);
        if (from) {
            copy_from_user_safe(gva, (const void *) from, len);
            from += len;
        }
        gpa_array[i] = (uint64_t) virt_to_phys(gva);
        size -= len;
    }

    return (uint64_t) virt_to_phys(gpa_array);
}


static uint64_t user_to_gpa(uint64_t from, uint32_t size) {
    if (size > QCU_KMALLOC_MAX_SIZE)
        return user_to_gpa_large(from, size);
    else if (size > 0)
        return user_to_gpa_small(from, size);
    else
        return from;
}

static void kfree_gpa(uint64_t pa, uint32_t size) {
    uint64_t *gpa_array;
    uint32_t i, len;

    if (size > QCU_KMALLOC_MAX_SIZE) {
        ptrace("large\n");
        gpa_array = (uint64_t *) phys_to_virt((phys_addr_t) pa);
        for (i = 0; size > 0; i++) {
            len = MIN(size, QCU_KMALLOC_MAX_SIZE);
            ptrace("i= %u, len= %u, pa= %p\n", i, len, (void *) gpa_array[i]);
            kfree(phys_to_virt((phys_addr_t) gpa_array[i]));
            size -= len;
        }
    }
    ptrace("phys= %p, virt= %p\n", (void *) pa, phys_to_virt((phys_addr_t) pa));
    kfree(phys_to_virt((phys_addr_t) pa));
}

// Send VirtuiHMCmd to virtio device
// @req: struct include command and arguments
// if the function is corrent, it return 0. otherwise, is -1
static int qcu_misc_send_cmd(VirtioQCArg *req) {
    struct scatterlist *sgs[2], req_sg, res_sg;
    unsigned int len;
    int err;
    VirtioQCArg *res;

    res = kmalloc_safe(sizeof(VirtioQCArg));
    memcpy(res, req, sizeof(VirtioQCArg));

    sg_init_one(&req_sg, req, sizeof(VirtioQCArg));
    sg_init_one(&res_sg, res, sizeof(VirtioQCArg));

    sgs[0] = &req_sg;
    sgs[1] = &res_sg;

    spin_lock(&qcu->lock);

    err = virtqueue_add_sgs(qcu->vq, sgs, 1, 1, req, GFP_ATOMIC);
    if (err) {
        error("virtqueue_add_sgs failed\n");
        goto out;
    }

    if (unlikely(!virtqueue_kick(qcu->vq))) {
        error("unlikely happen\n");
        goto out;
    }

    // TODO: do not use busy waiting
    while (!virtqueue_get_buf(qcu->vq, &len) && !virtqueue_is_broken(qcu->vq))
        cpu_relax();

    out:
    spin_unlock(&qcu->lock);

    memcpy(req, res, sizeof(VirtioQCArg));
    kfree(res);

    return err;
}

static struct virtio_qc_page *find_page_group(unsigned long addr, struct virtio_qc_mmap *priv) {
    struct list_head *tmp = NULL;
    struct virtio_qc_page *group = NULL;
    struct list_head *next_p;

    list_for_each_safe(tmp, next_p, &(priv->head))
    {
        group = list_entry(tmp,
        struct virtio_qc_page, list);
        if (group->uvm_start <= addr && group->uvm_end > addr) {
            return group;
        }
    }
    return NULL;
}

static uint64_t get_gpa_array(struct virtio_qc_page *group, int numOfblocks) {
    uint64_t *gpa_array;
    int i;
    gpa_array =
            (uint64_t *) kmalloc_safe(numOfblocks * sizeof(uint64_t));

    for (i = 0; i < numOfblocks; i++)
        gpa_array[i] = __pa(group->page[i]);

    return (uint64_t) virt_to_phys(gpa_array);
}



////////////////////////////////////////////////////////////////////////////////
///	Module & Execution control
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaRegisterFatBinary(VirtioQCArg *arg) {    // no extra parameters
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaUnregisterFatBinary(VirtioQCArg *arg) {    // no extra parameters
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaRegisterFunction(VirtioQCArg *arg) {    // pA: fatBin
    // pB: functrion name
    pfunc();
    ptrace("function name= %s\n", (char *) arg->pB);

    arg->pA = user_to_gpa(arg->pA, arg->pASize);
    arg->pB = user_to_gpa(arg->pB, arg->pBSize);


    qcu_misc_send_cmd(arg);

    kfree_gpa(arg->pA, arg->pASize);
    kfree_gpa(arg->pB, arg->pBSize);
}

void qcu_cudaRegisterVar(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}


void qcu_cudaLaunch(VirtioQCArg *arg) {    // pA: cuda kernel configuration
    // pB: cuda kernel parameters
    pfunc();

    arg->pA = user_to_gpa(arg->pA, arg->pASize);
    arg->pB = user_to_gpa(arg->pB, arg->pBSize);

    qcu_misc_send_cmd(arg);

    kfree_gpa(arg->pA, arg->pASize);
    kfree_gpa(arg->pB, arg->pBSize);
}

////////////////////////////////////////////////////////////////////////////////
///	Memory Management
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaMalloc(VirtioQCArg *arg) {    // pA: pointer of devPtr
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaMemset(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

void qcu_cudaFree(VirtioQCArg *arg) {    // pA: devPtr address
    pfunc();

    qcu_misc_send_cmd(arg);
}

static uint64_t get_gpa_array_start_phys(uint64_t from, struct virtio_qc_mmap *priv, uint32_t size, uint32_t *offset) {
    struct virtio_qc_page *group;
    unsigned int block_size;
    unsigned int start, start_offset, i;
    uint64_t *gpa_array;
    uint32_t rsize, len;

    group = find_page_group(from, priv);
    if (group == NULL) return (uint64_t) - 1;


    *offset = from - group->uvm_start;
    block_size = priv->block_size;
    start = *offset / block_size;
    start_offset = *offset % block_size;
    gpa_array =
            (uint64_t *) kmalloc_safe((group->numOfblocks - start) * sizeof(uint64_t));

    gpa_array[0] = __pa(group->page[start] + start_offset);

    rsize = block_size - start_offset;
    len = MIN(size, rsize);
    size -= len;

    for (i = 0; size > 0; i++) {
        gpa_array[i + 1] = __pa(group->page[start + 1 + i]);
        len = MIN(size, block_size);
        size -= len;
    }

    return (uint64_t) virt_to_phys(gpa_array);
}

static void free_gpa_array(uint64_t pa) {
    kfree(phys_to_virt((phys_addr_t) pa));
}

/*
void user_kernel_copy_send_cmd(VirtioQCArg *arg, uint64_t addr, uint32_t size)
{
	//arg->pB = user_to_gpa(arg->pB, arg->pBSize); // host
	//qcu_misc_send_cmd(arg);
	//kfree_gpa(arg->pB, arg->pBSize);
	arg->para = 1;
	arg->rnd = user_to_gpa(addr, size); // host
	qcu_misc_send_cmd(arg);
	kfree_gpa(arg->rnd, size);

}*/


void qcu_cudaMemcpy(VirtioQCArg *arg, void *dev) {
//#ifdef USER_KERNEL_COPY
    void *u_dst = NULL;
//#endif
    struct virtio_qc_mmap *priv;
    uint64_t gasp;
    priv = dev;
    pfunc();


    if (arg->flag == 1) // cudaMemcpyHostToDevice
    {
        //src: arg->pB
        //length: arg->pBSize
        //dst: arg->pA
        //fd: arg->pASize
        ptrace("pA= %p ,size= %u ,pB= %p, size= %u ,kind= %s\n",
               (void *) arg->pA, arg->pASize, (void *) arg->pB, arg->pBSize, "H2D");

#ifdef USER_KERNEL_COPY
        if(arg->pBSize > QCU_KMALLOC_MAX_SIZE)
        {
#endif
        gasp = get_gpa_array_start_phys(arg->pB, priv, arg->pBSize, &(arg->pASize));
        if (gasp == (uint64_t) - 1) {

            arg->para = 1;
            arg->pB = user_to_gpa(arg->pB, arg->pBSize); // host


            qcu_misc_send_cmd(arg);

            kfree_gpa(arg->pB, arg->pBSize);
        } else {
            arg->pB = gasp;
            qcu_misc_send_cmd(arg);
            free_gpa_array(arg->pB);
        }
#ifdef USER_KERNEL_COPY
        }
        else
        {
            //arg->pA is device pointer
            //user_kernel_copy_send_cmd(arg);
            arg->pB = user_to_gpa(arg->pB, arg->pBSize); // host
            qcu_misc_send_cmd(arg);
            kfree_gpa(arg->pB, arg->pBSize);
        }
#endif
    } else if (arg->flag == 2) // cudaMemcpyDeviceToHost
    {
        //dst: arg->pA
        //src: arg->pB
        //length: arg->pASize
        //fd: arg->pBSize
        ptrace("pA= %p ,size= %u ,pB= %p, size= %u ,kind= %s\n",
               (void *) arg->pA, arg->pASize, (void *) arg->pB, arg->pBSize, "D2H");
#ifdef USER_KERNEL_COPY
        if(arg->pASize > QCU_KMALLOC_MAX_SIZE)
        {
#endif
        gasp = get_gpa_array_start_phys(arg->pA, priv, arg->pASize, &(arg->pBSize));

        //group 		= find_page_group(arg->pA, dev);
        //arg->pA		= arg->pA - group->uvm_start; //offset
        //arg->pBSize	= group->file;
        //arg->pBSize: offset

        if (gasp == (uint64_t) - 1) {
            arg->para = 1;
            u_dst = (void *) arg->pA;
            arg->pA = user_to_gpa(0, arg->pASize); // host
            //arg->pB is device pointer

            qcu_misc_send_cmd(arg);

            gpa_to_user(u_dst, arg->pA, arg->pASize);
            kfree_gpa(arg->pA, arg->pASize);
        } else {
            arg->pA = gasp;
            qcu_misc_send_cmd(arg);
            free_gpa_array(arg->pA);
        }

#ifdef USER_KERNEL_COPY
        }
        else
        {
            u_dst = (void*)arg->pA;
            arg->pA = user_to_gpa( 0, arg->pASize); // host
            //arg->pB is device pointer
            qcu_misc_send_cmd(arg);

            gpa_to_user(u_dst, arg->pA, arg->pASize);
            kfree_gpa(arg->pA, arg->pASize);
        }
#endif
    } else if (arg->flag == 3) // cudaMemcpyDeviceToDevice
    {
        ptrace("pA= %p ,size= %u ,pB= %p, size= %u ,kind= %s\n",
               (void *) arg->pA, arg->pASize, (void *) arg->pB, arg->pBSize, "D2D");

        //arg->pA is device pointer
        //arg->pB is device pointer
        qcu_misc_send_cmd(arg);
    }
//	printk("========================\n"); //cocotion test time
}

void qcu_cudaMemcpyAsync(VirtioQCArg *arg, void *dev) {
    uint64_t gasp;
    struct virtio_qc_mmap *priv;
    priv = dev;

    if (arg->flag == 1) // cudaMemcpyHostToDevice
    {
        gasp = get_gpa_array_start_phys(arg->pB, priv, arg->pBSize, &(arg->pASize));
        arg->pB = gasp;
        qcu_misc_send_cmd(arg);
        free_gpa_array(arg->pB);
    } else if (arg->flag == 2) {
        gasp = get_gpa_array_start_phys(arg->pA, priv, arg->pASize, &(arg->pBSize));
        arg->pA = gasp;
        qcu_misc_send_cmd(arg);
        free_gpa_array(arg->pA);
    } else if (arg->flag == 3) {
        qcu_misc_send_cmd(arg);
    }


}

/*
void qcu_cudaMemcpyAsync(VirtioQCArg *arg, void *dev)
{

	struct virtio_qc_mmap *priv; //cocotion
	struct virtio_qc_page *group;
	priv = dev;

	if( arg->flag == 1 ) // cudaMemcpyHostToDevice
	{
			//TODO: if not find
		group = find_page_group(arg->pB, priv);
		priv->group = group;

		arg->para = priv->block_size*priv->group->numOfblocks;	

		arg->pB = arg->pB - group->uvm_start; //offset 
		arg->pASize = group->file; //fd
		
		qcu_misc_send_cmd(arg);
	}
	else if( arg->flag == 2 ) // cudaMemcpyDeviceToHost
	{
			//TODO: if not find
		group = find_page_group(arg->pA, priv);
		priv->group = group;

		arg->para = priv->block_size*priv->group->numOfblocks;	

		arg->pA = arg->pA - group->uvm_start; //offset 
		arg->pBSize = group->file; //fd
	
		qcu_misc_send_cmd(arg);
	}
	else if(arg->flag == 3 ) // cudaMemcpyDeviceToDevice
	{
		//arg->pA is device pointer
		//arg->pB is device pointer
		qcu_misc_send_cmd(arg);
	
	}
}
*/

////////////////////////////////////////////////////////////////////////////////
///	Device Management
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaGetDevice(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaGetDeviceCount(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaSetDevice(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaGetDeviceProperties(VirtioQCArg *arg) {
    void *prop;

    pfunc();

    prop = (void *) arg->pA;
    arg->pA = user_to_gpa(0, arg->pASize);

    qcu_misc_send_cmd(arg);

    gpa_to_user(prop, arg->pA, arg->pASize);
    kfree_gpa(arg->pA, arg->pASize);
}

void qcu_cudaDeviceSynchronize(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaDeviceReset(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaDeviceSetLimit(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

////////////////////////////////////////////////////////////////////////////////
///	Version Management
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaDriverGetVersion(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaRuntimeGetVersion(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

////////////////////////////////////////////////////////////////////////////////
///	Event Management
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaEventCreate(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaEventCreateWithFlags(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

void qcu_cudaEventRecord(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaEventSynchronize(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaEventElapsedTime(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

void qcu_cudaEventDestroy(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Error Handling
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaGetLastError(VirtioQCArg *arg) {
    pfunc();

    qcu_misc_send_cmd(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// Thread Management
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaThreadSynchronize(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

////////////////////////////////////////////////////////////////////////////////
///	new cuda API (for cuda runtime 9.0
////////////////////////////////////////////////////////////////////////////////

void qcu_cudaDeviceCanAccessPee(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

void qcu_cudaFuncGetAttributes(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

void qcu_cudaFuncSetAttribute(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}


////////////////////////////////////////////////////////////////////////////////
///	basic function
////////////////////////////////////////////////////////////////////////////////

static int qcu_misc_write(VirtioQCArg *arg) {    // pA: user src pointer and size
    arg->pA = user_to_gpa(arg->pA, arg->pASize);
    qcu_misc_send_cmd(arg);
    kfree_gpa(arg->pA, arg->pASize);
    return arg->cmd;
}

static int qcu_misc_read(VirtioQCArg *arg) {    // pA: user buffer
    void *u_dst = NULL;

    arg->pA = user_to_gpa(0, arg->pASize);
    qcu_misc_send_cmd(arg);
    gpa_to_user(u_dst, arg->pA, arg->pASize);
    kfree_gpa(arg->pA, arg->pASize);

    return arg->cmd;
}

static int free_page_blocks(struct virtio_qc_page *group, unsigned int order) {
    unsigned int numOfblocks = group->numOfblocks;
    int i;
    for (i = 0; i < numOfblocks; i++)
        free_pages(group->page[i], order);

    return 0;
}

static int free_host_page_blocks(VirtioQCArg *arg, struct virtio_qc_page *group) {
    unsigned int numOfblocks = group->numOfblocks;
    arg->pB = get_gpa_array(group, numOfblocks);
    arg->pBSize = numOfblocks;
    arg->cmd = VIRTQC_CMD_MUNMAP;
    qcu_misc_send_cmd(arg);

    kfree(phys_to_virt((phys_addr_t)(arg->pB)));

    return 0;
}

static int remove_host_map_file(VirtioQCArg *arg, struct virtio_qc_page *group) {
    arg->pA = group->uvm_start;
    arg->pBSize = group->file;
    arg->cmd = VIRTQC_CMD_MMAPRELEASE;
    qcu_misc_send_cmd(arg); //TO DO: safely check

    return 0;
}

static int qcu_misc_mmaprelease(VirtioQCArg *arg) {
    //uvm: arg->pA
    //arg->pB: filp->private_data

    struct virtio_qc_mmap *priv = (void *) (arg->pB);

    struct list_head *tmp = NULL;
    struct virtio_qc_page *group = NULL;
    struct list_head *next_p;

    unsigned long vmstart = arg->pA;

    //TODO: combine some code with virtqc_device_munmap
    list_for_each_safe(tmp, next_p, &(priv->head))
    {
        group = list_entry(tmp,
        struct virtio_qc_page, list);
        if (group->uvm_start == vmstart) {
            if (group->file != -1) {
                free_host_page_blocks(arg, group);
                remove_host_map_file(arg, group);
            }

            free_page_blocks(group, priv->order);

            list_del(&group->list);
            kfree(group->page);
            kfree(group);
            //arg->cmd = -1;
            //break;
            return 0;
        }
    }

    arg->cmd = -1;
    return 1;
}

static int mmapctl(struct virtio_qc_mmap *priv) {
    VirtioQCArg *arg;
    arg = kmalloc_safe(sizeof(VirtioQCArg));

    arg->cmd = VIRTQC_CMD_MMAPCTL;

    arg->pBSize = priv->group->numOfblocks * priv->block_size;

    arg->pB = priv->group->uvm_start;

    qcu_misc_send_cmd(arg);

    priv->group->file = arg->pA;

    if ((int) (arg->pA) == -1)
        goto err_open;

    kfree(arg);

    return 0;
    err_open:
    kfree(arg);
    return -EBADF;
}

static void qcummap(struct virtio_qc_mmap *priv) {
    VirtioQCArg *arg;
    arg = kmalloc_safe(sizeof(VirtioQCArg));

    arg->cmd = VIRTQC_CMD_MMAP;

    arg->pA = priv->group->file;
    arg->pASize = priv->group->numOfblocks;
    arg->pB = get_gpa_array(priv->group, arg->pASize);

    qcu_misc_send_cmd(arg);

    kfree(phys_to_virt((phys_addr_t)(arg->pB)));

    kfree(arg);
}

/////////zero-copy/////////

void qcu_cudaHostRegister(VirtioQCArg *arg, struct virtio_qc_mmap *priv) {
    uint64_t gasp;
    struct virtio_qc_page *group;

    gasp = get_gpa_array_start_phys(arg->pA, priv, arg->pASize, &(arg->pBSize));


    //fix for cudaHostGetDevicePointer start
    //TODO: if not find
    group = find_page_group(arg->pA, priv);
    priv->group = group;

    arg->pB = priv->block_size * priv->group->numOfblocks;

    //TODO: error handling
    mmapctl(priv);
    qcummap(priv);

    arg->rnd = arg->pA - group->uvm_start; //offset
    arg->pBSize = group->file; //fd
    //fix for cudaHostGetDevicePointer end

    arg->pA = gasp;
    qcu_misc_send_cmd(arg);
    free_gpa_array(arg->pA);

    group->data = arg->rnd; //fix for cudaHostGetDevicePointer
}

/*
void qcu_cudaHostRegister(VirtioQCArg *arg, struct virtio_qc_mmap *priv)
{
	struct virtio_qc_page *group;

	//TODO: if not find
	group = find_page_group(arg->pA, priv);
	priv->group = group;

	arg->pB = priv->block_size*priv->group->numOfblocks;

	//TODO: error handling	
	mmapctl(priv);
	qcummap(priv);

	arg->pA = arg->pA - group->uvm_start; //offset 
	arg->pBSize = group->file; //fd

	ptrace("group->uvm_start = %lx\n", group->uvm_start);

	qcu_misc_send_cmd(arg);

	group->data = arg->pA;	
}
*/

void qcu_cudaHostGetDevicePointer(VirtioQCArg *arg, struct virtio_qc_mmap *priv) {
    struct virtio_qc_page *group;
    //TODO: if not find
    group = find_page_group(arg->pB, priv);
    priv->group = group;

    arg->pB = group->data;

    qcu_misc_send_cmd(arg);
}

void qcu_cudaHostUnregister(VirtioQCArg *arg, struct virtio_qc_mmap *priv) {
    struct virtio_qc_page *group;
    uint32_t size;
    uint64_t gasp;
    group = find_page_group(arg->pA, priv);
    size = group->numOfblocks * priv->block_size;


    gasp = get_gpa_array_start_phys(arg->pA, priv, size, &(arg->pBSize));

    arg->pA = gasp;
    arg->pASize = size;
    qcu_misc_send_cmd(arg);
    free_gpa_array(arg->pA);
}

/*
void qcu_cudaHostUnregister(VirtioQCArg *arg, struct virtio_qc_mmap *priv)
{
	struct virtio_qc_page *group;

	//TODO: if not find
	group = find_page_group(arg->pA, priv);
	priv->group = group;

	arg->pB = group->data;
	qcu_misc_send_cmd(arg);
}
*/

void qcu_cudaSetDeviceFlags(VirtioQCArg *arg) {
    ptrace("cocotion qcu_cudaSetDeviceFlags start ok\n");
    qcu_misc_send_cmd(arg);
    ptrace("cocotion qcu_cudaSetDeviceFlags end ok\n");
}

void qcu_cudaFreeHost(VirtioQCArg *arg, struct virtio_qc_mmap *priv) {
    struct virtio_qc_page *group;
//	uint64_t p = arg->pA;	
//	qcu_cudaHostUnregister(arg, priv);

    //TODO: if not find
    group = find_page_group(arg->pA, priv);
    priv->group = group;

    arg->pA = group->uvm_start;
    arg->pB = (uint64_t)(priv);

    arg->cmd = VIRTQC_CMD_MMAPRELEASE;

    if (!qcu_misc_mmaprelease(arg))
        arg->cmd = 0;
}

//stream
void qcu_cudaStreamCreate(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

void qcu_cudaStreamDestroy(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}

void qcu_cudaStreamSynchronize(VirtioQCArg *arg) {
    qcu_misc_send_cmd(arg);
}


// @_cmd: device command
// @_arg: argument of cuda function
// this function reture cudaError_t.
static long qcu_misc_ioctl(struct file *filp, unsigned int _cmd, unsigned long _arg) {
    VirtioQCArg *arg;
    int err;

    arg = kmalloc_safe(sizeof(VirtioQCArg));
    copy_from_user_safe(arg, (void *) _arg, sizeof(VirtioQCArg));
    //ptrace("_arg= %p, arg= %p\n", (void*)_arg, arg);

    arg->cmd = _cmd;
//	get_random_bytes(&arg->rnd, sizeof(int));

    switch (arg->cmd) {
        case VIRTQC_CMD_WRITE:
            err = (int) qcu_misc_write(arg);
            break;

        case VIRTQC_CMD_READ:
            err = (int) qcu_misc_read(arg);
            break;

            //case VIRTQC_CMD_MMAPCTL:
            //arg->pA = (uint64_t)(filp->private_data);
            //qcu_misc_mmapctl(arg);
            //break;

        case VIRTQC_CMD_MMAPRELEASE:
            arg->pB = (uint64_t)(filp->private_data);
            qcu_misc_mmaprelease(arg);
            break;

            // Module & Execution control (driver API)
        case VIRTQC_cudaRegisterFatBinary:
            qcu_cudaRegisterFatBinary(arg);
            break;

        case VIRTQC_cudaUnregisterFatBinary:
            qcu_cudaUnregisterFatBinary(arg);
            break;

        case VIRTQC_cudaRegisterFunction:
            qcu_cudaRegisterFunction(arg);
            break;

        case VIRTQC_cudaRegisterVar:
            qcu_cudaRegisterVar(arg);
            break;

        case VIRTQC_cudaLaunch:
            qcu_cudaLaunch(arg);
            break;

            // Memory Management (runtime API)
        case VIRTQC_cudaMalloc:
            qcu_cudaMalloc(arg);
            break;

        case VIRTQC_cudaMemset:
            qcu_cudaMemset(arg);
            break;

        case VIRTQC_cudaMemcpy:
            qcu_cudaMemcpy(arg, filp->private_data);
            break;

        case VIRTQC_cudaMemcpyAsync:
            qcu_cudaMemcpyAsync(arg, filp->private_data);
            break;

        case VIRTQC_cudaFree:
            qcu_cudaFree(arg);
            break;

            // Device Management (runtime API)
        case VIRTQC_cudaGetDevice:
            qcu_cudaGetDevice(arg);
            break;

        case VIRTQC_cudaGetDeviceCount:
            qcu_cudaGetDeviceCount(arg);
            break;

        case VIRTQC_cudaSetDevice:
            qcu_cudaSetDevice(arg);
            break;

        case VIRTQC_cudaGetDeviceProperties:
            qcu_cudaGetDeviceProperties(arg);
            break;

        case VIRTQC_cudaDeviceSynchronize:
            qcu_cudaDeviceSynchronize(arg);
            break;

        case VIRTQC_cudaDeviceReset:
            qcu_cudaDeviceReset(arg);
            break;

        case VIRTQC_cudaDeviceSetLimit:
            qcu_cudaDeviceSetLimit(arg);
            break;

            //case VIRTQC_checkCudaCapabilities:
            //	qcu_checkCudaCapabilities(arg);
            //	break;
            ///////////////////////////////////
            // Version Management (runtime API)
        case VIRTQC_cudaDriverGetVersion:
            qcu_cudaDriverGetVersion(arg);
            break;

        case VIRTQC_cudaRuntimeGetVersion:
            qcu_cudaRuntimeGetVersion(arg);
            break;

            // Event Management (runtime API)
        case VIRTQC_cudaEventCreate:
            qcu_cudaEventCreate(arg);
            break;

        case VIRTQC_cudaEventCreateWithFlags:
            qcu_cudaEventCreateWithFlags(arg);
            break;

        case VIRTQC_cudaEventRecord:
            qcu_cudaEventRecord(arg);
            break;

        case VIRTQC_cudaEventSynchronize:
            qcu_cudaEventSynchronize(arg);
            break;

        case VIRTQC_cudaEventElapsedTime:
            qcu_cudaEventElapsedTime(arg);
            break;

        case VIRTQC_cudaEventDestroy:
            qcu_cudaEventDestroy(arg);
            break;

            // Error Handling (runtime API)
        case VIRTQC_cudaGetLastError:
            qcu_cudaGetLastError(arg);
            break;

            //Zero-copy
        case VIRTQC_cudaHostRegister:
            qcu_cudaHostRegister(arg, filp->private_data);
            break;

        case VIRTQC_cudaHostGetDevicePointer:
            qcu_cudaHostGetDevicePointer(arg, filp->private_data);
            break;

        case VIRTQC_cudaHostUnregister:
            qcu_cudaHostUnregister(arg, filp->private_data);
            break;

        case VIRTQC_cudaSetDeviceFlags:
            qcu_cudaSetDeviceFlags(arg);
            break;

        case VIRTQC_cudaFreeHost:
            qcu_cudaFreeHost(arg, filp->private_data);
            break;

            //stream
        case VIRTQC_cudaStreamCreate:
            qcu_cudaStreamCreate(arg);
            break;

        case VIRTQC_cudaStreamDestroy:
            qcu_cudaStreamDestroy(arg);
            break;

        case VIRTQC_cudaStreamSynchronize:
            qcu_cudaStreamSynchronize(arg);
            break;


            // Thread Management
        case VIRTQC_cudaThreadSynchronize:
            qcu_cudaThreadSynchronize(arg);
            break;

            ///////////////////////////////////////////////////////////////
            ///	new cuda API (for cuda runtime 9.0
            ///////////////////////////////////////////////////////////////

        case VIRTQC_cudaDeviceCanAccessPeer:
            qcu_cudaDeviceCanAccessPee(arg);
            break;

        case VIRTQC_cudaFuncGetAttributes:
            qcu_cudaFuncGetAttributes(arg);
            break;
        case VIRTQC_cudaFuncSetAttribute:
            qcu_cudaFuncSetAttribute(arg);
            break;

        default:
        error("unknow cmd= %d\n", arg->cmd);
            break;
    }

    copy_to_user_safe((void *) _arg, arg, sizeof(VirtioQCArg));

    kfree(arg);
    ptrace("return\n");
    return 0;
}

static int qcu_misc_open(struct inode *inode, struct file *filp) {
    VirtioQCArg *arg;
    struct virtio_qc_mmap *priv;
    arg = kmalloc_safe(sizeof(VirtioQCArg));

    arg->cmd = VIRTQC_CMD_OPEN;

    try_module_get(THIS_MODULE);

    priv = kmalloc(sizeof(struct virtio_qc_mmap), GFP_KERNEL);
    filp->private_data = priv;
//	priv->file = kmalloc(sizeof(virtio_qc_file), GFP_KERNEL);
//	priv->block_id = 0;

    priv->block_size = PAGE_SIZE * 1024; //test 4096k
    INIT_LIST_HEAD(&(priv->head));

    arg->pASize = priv->block_size;
    qcu_misc_send_cmd(arg);

    //priv->hpid = arg->pA;
    //*(priv->file) = arg->pA;

    //kfree(arg);

    //if(*(priv->file) == -1)
    //	return -EBADF;

    return 0;
}

static int qcu_misc_release(struct inode *inode, struct file *filp) {

    //VirtioQCArg *arg;
    struct virtio_qc_mmap *priv;
    priv = filp->private_data;

    //arg = kmalloc_safe(sizeof(VirtioQCArg));

    //arg->pA	= *(priv->file);

    //arg->cmd = VIRTQC_CMD_CLOSE;

    //qcu_misc_send_cmd(arg);

    //if ((int)arg->pB == -1)
    //	printk(KERN_ERR "virtio-hm: virthm_release close error\n");

//	kfree(priv->file);
    kfree(priv);
    module_put(THIS_MODULE);

    //kfree(arg);

    return 0;
}

static void virtqc_device_mmap(struct vm_area_struct *vma) {
/*
	struct virtio_qc_mmap *priv = vma->vm_private_data;
	VirtioQCArg *arg;
	int i;

	ptrace("mmap start\n");
	
	arg = kmalloc_safe(sizeof(VirtioQCArg));
	
	arg->cmd = VIRTQC_CMD_MMAP;
	ptrace("mmap command = %d\n", arg->cmd);
	
	arg->pASize	= priv->block_size;		
	arg->pA		= priv->group->file;

	for(i = 0; i < priv->group->numOfblocks; i++)
	{
		arg->pB = __pa((priv->group->page)[i]);
		
		arg->pBSize = i*priv->block_size; //offset
		qcu_misc_send_cmd(arg); //cocotion test, next step test this unmark
	}
	
	kfree(arg);
*/
    ptrace("mmap ok\n");

}

static void virtqc_device_munmap(struct vm_area_struct *vma) {
    struct virtio_qc_mmap *priv = vma->vm_private_data;
    VirtioQCArg *arg;

    struct list_head *tmp = NULL;
    struct virtio_qc_page *group = NULL;
    struct list_head *next_p;

    arg = kmalloc_safe(sizeof(VirtioQCArg));

    list_for_each_safe(tmp, next_p, &(priv->head))
    {
        group = list_entry(tmp,
        struct virtio_qc_page, list);

        if (group->file != -1) {
            free_host_page_blocks(arg, group);
            remove_host_map_file(arg, group);
        }

        free_page_blocks(group, priv->order);

        list_del(&group->list);
        kfree(group->page);
        kfree(group);
    }
//////////
    kfree(arg);
//	ptrace("munmap ok\n");
}

static struct vm_operations_struct virtqc_mmap_ops = {
        .open  = virtqc_device_mmap,
        .close = virtqc_device_munmap,
};

static int compare(const void *lhs, const void *rhs) {
    int lhs_integer = *(const unsigned long *) (lhs);
    int rhs_integer = *(const unsigned long *) (rhs);

    if (lhs_integer < rhs_integer) return -1;
    if (lhs_integer > rhs_integer) return 1;
    return 0;
}

static int qcu_misc_mmap(struct file *filp, struct vm_area_struct *vma) {

    int i;
    unsigned int block_size, order, numOfblocks, size;

    struct virtio_qc_mmap *priv = filp->private_data;
    struct virtio_qc_page *group;

    size = vma->vm_end - vma->vm_start;

    block_size = priv->block_size;
    order = get_order(block_size);
    //max order = 11
    priv->order = order;
    numOfblocks = size / block_size;

    vma->vm_private_data = (void *) priv;

    group = kmalloc(sizeof(struct virtio_qc_page), GFP_KERNEL);
    priv->group = group;

    list_add(&(group->list), &(priv->head));

    group->numOfblocks = numOfblocks;
    group->uvm_start = vma->vm_start;
    group->uvm_end = vma->vm_end;

    group->file = -1;

    group->page = kmalloc(sizeof(unsigned long) * numOfblocks, GFP_KERNEL);

    for (i = 0; i < numOfblocks; i++) {
        unsigned long page = __get_free_pages(GFP_KERNEL, order);
        if (!page) return ENOMEM;
        group->page[i] = page;
    }

    sort(group->page, numOfblocks, sizeof(unsigned long), &compare, NULL);

    for (i = 0; i < numOfblocks; i++) {
        vma->vm_pgoff = __pa(group->page[i]) >> PAGE_SHIFT;

        if (remap_pfn_range(vma, vma->vm_start + i * block_size, vma->vm_pgoff,
                            block_size,
                            vma->vm_page_prot)) {
            free_pages(group->page[i], order);
            kfree(group->page);
            return -EAGAIN;
        }
    }

//	if((i=mmapctl(priv))==-1)
//		return -EBADF;

    vma->vm_ops = &virtqc_mmap_ops;
    virtqc_device_mmap(vma);

    return 0;
}


struct file_operations qcu_misc_fops = {
        .owner        = THIS_MODULE,
        .open            = qcu_misc_open,
        .release        = qcu_misc_release,
        .unlocked_ioctl = qcu_misc_ioctl,
        .mmap            = qcu_misc_mmap,
};

static struct miscdevice qcu_misc_driver = {
        .minor = MISC_DYNAMIC_MINOR,
        .name  = "qcuda",
        .fops  = &qcu_misc_fops,
};

//####################################################################
//   virtio operations
//####################################################################

static void qcu_virtio_cmd_vq_cb(struct virtqueue *vq) {
    /*
       VirtioQCArg *cmd;
       unsigned int len;

       while( (cmd = virtqueue_get_buf(vq, &len))!=NULL ){
       ptrace("read cmd= %d , rnd= %d\n", cmd->cmd, cmd->rnd);
       }
     */
}

/*
   static int qcu_virtio_init_vqs()
   {
   stract virtqueue *vqs[2];
   vq_callback_t *cbs[] = { qcu_virtio_in_vq_cb, qcu_virtio_out_vq_cb };
   const char *names[] = { "input_handle", "output_handle" };
   int err;

   err = qcu->vdev->config->find_vqs(qcu->vdev, 2, vqs, cbs, names);
   if( err ){
   ptrace("find_vqs failed.\n");
   return err;
   }

   qcu->in_vq = vqs[0];
   qcu->out_vq= vqs[1];

   return 0;
   }

   static int qcu_virtio_remove_vqs()
   {
   qcu->vdev->config->del_vqs(qcu->vdev);
   kfree(qcu->in_vq);
   kfree(qcu->out_vq);
   }
 */
static int qcu_virtio_probe(struct virtio_device *vdev) {
    int err;

    qcu = kzalloc_safe(sizeof(struct virtio_qcuda));
    if (!qcu) {
        err = -ENOMEM;
        goto err_kzalloc;
    }

    vdev->priv = qcu;
    qcu->vdev = vdev;

    qcu->vq = virtio_find_single_vq(vdev, qcu_virtio_cmd_vq_cb,
                                    "request_handle");
    if (IS_ERR(qcu->vq)) {
        err = PTR_ERR(qcu->vq);
        error("init vqs failed.\n");
        goto err_init_vq;
    }

    err = misc_register(&qcu_misc_driver);
    if (err) {
        error("virthm: register misc device failed.\n");
        goto err_reg_misc;
    }

    spin_lock_init(&qcu->lock);

    return 0;

    err_reg_misc:
    vdev->config->del_vqs(vdev);
    err_init_vq:
    kfree(qcu);
    err_kzalloc:
    return err;
}

static void qcu_virtio_remove(struct virtio_device *vdev) {
//    int err;

//    err = misc_deregister(&qcu_misc_driver);
    misc_deregister(&qcu_misc_driver);
//    if (err) {
//        error("misc_deregister failed\n");
//    }
    qcu->vdev->config->reset(qcu->vdev);
    qcu->vdev->config->del_vqs(qcu->vdev);
//    kfree(qcu->vq);  // the 'del_vqs' will kfree the qcu->vq
    kfree(qcu);
}

static unsigned int features[] = {};

static struct virtio_device_id id_table[] = {
        {VIRTIO_ID_QC, VIRTIO_DEV_ANY_ID},
        {0},
};

static struct virtio_driver virtio_qcuda_driver = {
        .feature_table      = features,
        .feature_table_size = ARRAY_SIZE(features),
        .driver.name        = KBUILD_MODNAME,
        .driver.owner       = THIS_MODULE,
        .id_table           = id_table,
        .probe              = qcu_virtio_probe,
        .remove             = qcu_virtio_remove,
};

static int __init

init(void) {
    int ret;

    ret = register_virtio_driver(&virtio_qcuda_driver);
    if (ret < 0) {
        error("register virtio driver faild (%d)\n", ret);
    }
    return ret;
}

static void __exit

fini(void) {
    unregister_virtio_driver(&virtio_qcuda_driver);
}

module_init(init);
module_exit(fini);

MODULE_DEVICE_TABLE(virtio, id_table
);
MODULE_DESCRIPTION("Qemu Virtio qCUdriver");
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Yu-Shiang Lin");
