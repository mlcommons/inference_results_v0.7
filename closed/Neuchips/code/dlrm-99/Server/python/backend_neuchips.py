#
# NEUCHIPS CONFIDENTIAL
#
# Copyright (C) 2018-2020 NEUCHIPS Corp.
# All Rights Reserved.
# Author: Tzu-Jen Julius Lo <tzujen_lo@neuchips.ai>
#         Brock Ko <brock_ko@neuchips.ai>
#
# The information and source code contained herein is the exclusive property
# of NEUCHIPS CORPORATION and may not be disclosed, examined or reproduced
# in whole or in part without explicit written authorization from the company.
#
import backend
from datetime import datetime
import math
import multiprocessing as mp
import numpy as np
import os
import queue
import torch

#
# NEUCHIPS proprietary packages
#
from utils_neuchips import RecAccel
from utils_neuchips import jprint
import utils_neuchips_lut

#
# RecAccel configurations
#
ACCELS_MAX_NUM = 2

#
# PCIE configurations
#
PCIE_VENDOR_ID = 0x1172
PCIE_PRODUCT_ID = 0xe003

#
# Global variables
#
rridx = mp.Value('i', 1)
#
accels = []
workers = []
#
wpool = None


#
# MlperfQuery - an object to submit a query to RecAccel worker
#
class MlperfQuery(object):

    def __init__(self, ndata, n_infs, ticket, buff=None, paral_dram=True):
        self.ndata = ndata
        self.n_infs = n_infs
        self.ticket = ticket
        self.buff = buff
        self.paral_dram = paral_dram


# predict_handler - target function of RecAccel worker
# @accel: associated RecAccel instance
#
# Return: none
def predict_handler(accel):

    def _predict(accel, bid, n_infs, fd=None):
        jprint("RecAccel-%d - __predict" % (accel.cid), "fd=0x%08x" % (fd))
        batch_size = min(n_infs, accel.batch_size)
        burst_len = math.ceil(n_infs/accel.batch_size)
        accel.predict(bid, batch_size, burst_len, fd)
        jprint("RecAccel-%d - __predict done" % (accel.cid))
        return

    fd = accel.clib.PCIE_Open(PCIE_VENDOR_ID, PCIE_PRODUCT_ID, accel.cid)

    # Main entry
    while accel.ready():
        try:
            mlq = accel.runq.get(block=True, timeout=0.01)
            binfo = accel.buffs_pool[mlq.buff]

            #
            # Before we start to predict, make sure previous user already
            # read out all the results
            #
            binfo.sync.wait()
            binfo.sync.clear()
            _predict(accel, binfo.bid, mlq.n_infs, fd)

            #
            # Return a buffer back to the pool; this allows another process
            # to write inputs simultaneously
            #
            if mlq.paral_dram:
                accel.buffs.put(mlq.buff)

            #
            accel.tickets_event[mlq.ticket].set()
            jprint("RecAccel-%d waking up...%d" % (accel.cid, mlq.ticket))

            #
            # Wake up a waiter if any
            #
            if accel.waitq.qsize():
                jprint("RecAccel-%d waitq get" % (accel.cid))
                todo = accel.waitq.get()
                accel.tickets[todo.ticket].set()

            jprint("RecAccel-%d next try" % (accel.cid))

        except queue.Empty:
            continue

    print("Error at RecAccel-%d, exiting..." % (accel.cid))
    accel.clib.PCIE_Close(fd)
    exit(accel.status)


#
# BackendRecAccel - NEUCHIPS RecAccel, a DLRM inference accelerating solution
#
# version() - version
# name() - name
# load() - load LUT to RecAccel DRAM
# predict() - RecAccel backend prediction entry function
#
# rr_get_accel() - select a RecAccel by round-robin
# do_predict() - major logic to interact with RecAccel worker/hardware
# collate_pre() - pre-process of collation
# collate_post() - collate dense and sparse torch to bytes
#
class BackendRecAccel(backend.Backend):

    def __init__(self):
        global accels
        global workers
        global wpool

        jprint("recAccel - __init__")

        for i in range(ACCELS_MAX_NUM):
            accel = RecAccel(cid=i, batch_size=16, freq_mhz=200,
                             paral_dram=True)
            if accel.ready() is False:
                print("init accel", i, "failed")
                continue

            # dedicated process for each accel
            procname = "RecAccel-" + str(i)
            worker = mp.Process(target=predict_handler, args=(accel,),
                                name=procname)
            worker.daemon = True
            workers.append(worker)

            accels.append(accel)
            print("RecAccel-%d @%d MHz" % (accel.cid, accel.freq_mhz),
                  "ping-pong =", len(accel.buffs_pool) == 2)

        #
        # RecAccel input/output arguments
        #
        self.input_q = 4
        self.output_q = 7

        self.nbyte_per_inf = 308
        self.nbyte_per_batch = self.nbyte_per_inf * 16

        #
        # RecAccel capability
        #
        self.max_batchsize = 1024
        #
        wpool = mp.Pool(4)
        print("Backend max batch size = %d" % (self.max_batchsize))

    def version(self):
        return "1.0"

    def name(self):
        return "NEUCHIPS-RecAccel"

    # load - load LUT to RecAccel DRAM
    # @model_path: path of model
    #
    # Return: none
    def load(self, model_path, inputs=None, outputs=None):
        global accels
        global workers

        def _write_embs(emb):
            conf = [[19, 10, 1, 23],
                    [0, 22, 4, 14, 13, 7, 17, 15, 24, 8, 25, 18, 12, 16, 5],
                    [21, 11, 2, 3],
                    [9, 20, 6]]

            for i in range(4):
                data = []

                jprint("    load DDR4%c" % (chr(ord('A')+i)))
                for idx in conf[i]:
                    data.extend(map(lambda f: f if f >= 0 else 256 + f,
                                    emb[idx].flip([1]).reshape(-1).tolist()))

                for accel in accels:
                    accel.dma_write(accel.mem_addr_ddr4[i], bytes(data),
                                    verify=False)
                del data

        jprint("recAccel - load")

        #
        # TODO: adaptively load LUT only when LUT is not loaded before
        #
        for worker in workers:
            worker.start()
            print(worker.name, worker.pid)
        return self

        emb, bot, top = utils_neuchips_lut.collate_lut(model_path)

        _write_embs(emb)

        for worker in workers:
            worker.start()
            print(worker.name, worker.pid)
        return self

    # predict - RecAccel backend prediction entry function
    # @ndata: NEUCHIPS collated data for RecAccel
    # @rsize: expected inference size
    #
    # Return: Predicted result in torch
    def predict(self, ndata, rsize):
        global wpool

        size = len(ndata) // self.nbyte_per_inf

        if size <= self.max_batchsize:
            return self.do_predict(ndata, size)

        #
        # Deploy multiprocessing pool
        #
        # @size = @quot * @self.max_batchsize + @resid
        #       = @quots + @resid
        quot = size // self.max_batchsize
        quots = quot * self.max_batchsize
        resid = size - quots
        # print("(%d) %d = %d + %d" %(rsize, size, quots, resid))
        nbyte_quots = quots * self.nbyte_per_inf

        _ndata = np.frombuffer(ndata[:nbyte_quots], dtype=np.byte)
        _ndata = list(map(bytes, np.array_split(_ndata, quot)))
        if resid != 0:
            _ndata.append(ndata[nbyte_quots:])

        jprint("pool mapping...")
        t = wpool.map(self.do_predict, _ndata)
        jprint("all done")
        res = torch.FloatTensor(rsize, 1)
        torch.cat(t, out=res)
        return res

    # rr_get_accel - select a RecAccel by round-robin
    #
    # Return: a RecAccel instance
    def rr_get_accel(self):
        global rridx
        global accels

        with rridx.get_lock():
            rridx.value += 1
            if rridx.value >= ACCELS_MAX_NUM:
                rridx.value = 0
            idx = rridx.value
        return accels[idx]

    # do_predict - major logic to interact with RecAccel worker/hardware
    # @ndata: NEUCHIPS collated data for RecAccel
    # @size: inference size to predict; if None, calculate the size from @ndata
    #
    # Return: predicted result in torch
    def do_predict(self, ndata, size=None):

        if size is None:
            size = len(ndata) // self.nbyte_per_inf

        accel = self.rr_get_accel()
        jprint("benchmark enter", accel.cid)

        #
        # Decrypt 16-infs aligned @ndata to build up splitting map
        #
        batch = len(ndata) // self.nbyte_per_batch
        npb = self.nbyte_per_batch
        trail = np.frombuffer(ndata[npb-1::npb], dtype=np.uint8)
        valid = np.subtract(np.ones(batch, dtype=np.uint8) * accel.batch_size,
                            trail)
        # build up list for split
        smap = np.empty(batch * 2, dtype=np.uint8)
        smap[0::2] = valid
        smap[1::2] = trail

        ticket = None
        while ticket is None:
            try:
                ticket = accel.tickets.get()
                tevent = accel.tickets_event[ticket]
            except queue.Empty:
                jprint("Failed, run out of tickets at RecAccel-%d"
                       % (accel.cid))
                continue
        jprint("benchmark - T")

        res = None
        while res is None:
            try:
                buff = accel.buffs.get()
                binfo = accel.buffs_pool[buff]
                jprint("benchmark - T1", "fd=0x%08x RecAccel-%d buff=%d" %
                       (accel.fd, accel.cid, binfo.bid))

                accel.dma_write(binfo.addr_in, ndata)

                mlq = MlperfQuery(None, size, ticket, buff, accel.paral_dram)
                accel.runq.put(mlq)
                jprint("benchmark runq waiting...%d (%d/%d)" %
                       (mlq.ticket, sum(valid), size))
                #
                # Wait for RecAccel completes associated prediction
                #
                tevent.wait()
                tevent.clear()

                #
                # Once prediction completes, RecAccel returns the buffer back
                # to the pool as soon as possible. So we don't need to worry
                # about buffer here, just put ticket off.
                #
                accel.tickets.put(ticket)

                res = accel.dma_read_res_in_torch(binfo.addr_out, size,
                                                  self.output_q)
                # allow writing to proceed
                binfo.sync.set()
                if not accel.paral_dram:
                    accel.buffs.put(buff)

                # extract valid inferences
                return torch.cat(res.split(smap.tolist())[0::2])

            except queue.Empty:
                mlq = MlperfQuery(ndata, size, ticket)
                accel.waitq.put(mlq)
                jprint("benchmark waitq waiting...%d" % (mlq.ticket))

                tevent.wait()
                tevent.clear()

    # collate_pre - pre-process of collation
    # @dense: torch of dense
    # @sparse: torch of sparse
    #
    # Return: pair of pre-processed torch
    def collate_pre(self, dense, sparse):
        dns = torch.clamp(torch.floor(dense.to(torch.float64) *
                                      pow(2, self.input_q) + 0.5),
                          0, 255).to(torch.uint8)
        sps = sparse.T
        return dns, sps

    # collate_post - collate dense and sparse torch to bytes
    # @d: pre-processed dense in torch
    # @s: pre-processed sparse in torch
    #
    # Return: collated metadata in bytes
    def collate_post(self, d, s):
        zerox16 = np.zeros(16, dtype=np.uint8)

        def __collate(d, s):
            res = []

            dt = d.T
            for inf in dt:
                res.extend(inf)
                res.extend(np.repeat(zerox16, 3))

            for inf in s:
                for i in range(26):
                    res.extend(list(int(inf[i].item()).to_bytes(8, 'little',
                                                                signed=False)))
                res.extend(np.repeat(zerox16, 3))

            return res

        # Entry
        infs = d.shape[0]
        assert(infs == s.shape[0])

        d_np = np.asarray(d, dtype=np.uint8)
        s_np = np.asarray(s, dtype=np.uint64)

        base = 0
        res = []
        _infs = infs
        r = 0
        while _infs > 0:
            if _infs < 16:
                r = 16 - _infs
                d_np = np.concatenate((d_np, np.zeros((r, 13),
                                                      dtype=np.uint8)))
                s_np = np.concatenate((s_np, np.zeros((r, 26),
                                                      dtype=np.uint64)))
            res.extend(__collate(d_np[base:base+16], s_np[base:base+16]))
            _infs -= 16
            base += 16

        # for i in range(0, 52, 4):
        #     print("0x%02x" %(i), res[i*16:i*16+16])

        res[-1] = r
        return bytes(res)
