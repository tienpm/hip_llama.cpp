import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

font = {"size": 12}
plt.rc("font", **font)

colors = [
    "tab:red",
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
styles = [
    "_",
    "o",
    "s",
    "v",
    "^",
    "D",
    ">",
    "<",
    "*",
    "h",
    "H",
    "+",
    "1",
    "2",
    "3",
    "4",
    "8",
    "p",
    "d",
    "|",
    ".",
    ",",
]

markersize = 10
markerwidth = 2
maxchar = 25


def roofline(
    plot_path, flops, hbm_ai, l2_ai=[], l1_ai=[], labels=[], flag="HBM", dtype="FP32"
):
    assert flops is not None
    # assert max(flops) == 0.0
    if (not hbm_ai) and (not l2_ai) and (not l1_ai):
        raise TypeError("AIHBM, AIL2 and AIL1 can not all be empty!")
    if (
        (len(flops) != len(hbm_ai))
        or (len(flops) != len(l2_ai))
        or (len(flops) != len(l1_ai))
    ):
        raise ValueError("FLOPS needs to have the same length as AI!")
    if flag != "HBM" and flag != "L2" and flag != "L1" and flag != "all":
        raise ValueError("flag needs to be one of HBM, L2, L1, and all!")
    labels = [x[:maxchar] for x in labels]

    mem_roofs = [("HBM ", 1.2 * pow(10, 12))]
    cmp_roofs = []
    if dtype == "FP32":
        cmp_roofs.append(("Peak FP32 (GFLOPs)", 23.1 * pow(10, 3)))
    elif dtype == "FP16":
        cmp_roofs.append(("Peak FP16 (GFLOPs)", 184.6 * pow(10, 3)))
    elif dtype == "FP64":
        cmp_roofs.append(("Peak FP64 (GFLOPs)", 11.5 * pow(10, 3)))
    elif dtype == "BF16":
        cmp_roofs.append(("Peak BF16 (GFLOPs)", 92.3 * pow(10, 3)))
    else:
        cmp_roofs.append(("Peak INT (GFLOPs)", 92.3))

    print(mem_roofs)
    print(cmp_roofs)
    fig = plt.figure(1, figsize=(10.67, 6.6))
    plt.clf()
    ax = fig.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    # AMD Instruction Roofline
    ax.set_xlabel("Instruction Intensity (FLOPs / Bytes)")
    ax.set_ylabel("Performance (GFLOPs)")

    # AMD Instruction Roofline
    nx = 10000
    xmin = -5
    xmax = 5
    ymin = 1e-1
    ymax = 50000

    ax.set_xlim(10**xmin, 10**xmax)
    ax.set_ylim(ymin, ymax)

    ixx = int(nx * 0.02)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scomp_x_elbow = []
    scomp_ix_elbow = []
    smem_x_elbow = []
    smem_ix_elbow = []

    x = np.logspace(xmin, xmax, nx)
    print(x)
    for i, roof in enumerate(cmp_roofs):
        for ix in range(1, nx):
            a = float(mem_roofs[0][1] * x[ix])
            b = roof[1] * 1024
            c = mem_roofs[0][1] * x[ix - 1]
            if a >= b > c:
                scomp_x_elbow.append(x[ix - 1])
                scomp_ix_elbow.append(ix - 1)
                break

    for roof in mem_roofs:
        for ix in range(1, nx):
            if roof[1] * x[ix] >= cmp_roofs[0][1] * 1024 > roof[1] * x[ix - 1]:
                smem_x_elbow.append(x[ix - 1])
                smem_ix_elbow.append(ix - 1)
                break

    # print(scomp_x_elbow)
    # print(scomp_ix_elbow)
    # print(smem_x_elbow)
    # print(smem_ix_elbow)

    # Plot Peak compute
    for i in range(len(cmp_roofs)):
        roof = cmp_roofs[i][1] * 1024
        y = np.ones(len(x)) * roof
        ax.plot(
            x[scomp_ix_elbow[i] :],
            y[scomp_ix_elbow[i] :],
            c=colors[i % len(colors)],
            ls="-",
            lw="2",
        )

    # Plot Peak memory
    for i in range(len(mem_roofs)):
        roof = mem_roofs[i][1]
        y = x * roof
        ax.plot(
            x[: smem_ix_elbow[i] + 1],
            y[: smem_ix_elbow[i] + 1],
            c=colors[i % len(colors)],
            ls="-",
            lw="2",
        )

    for i in range(len(hbm_ai)):
        if flag == "L1":
            ax.plot(
                float(l1_ai[i]),
                float(flops[i]),
                c=colors[i % len(colors)],
                marker=styles[0],
                linestyle="None",
                ms=markersize,
                markerfacecolor="none",
                markeredgewidth=markerwidth,
                label=labels[i] if labels else "unknown",
            )
        elif flag == "L2":
            ax.plot(
                float(l2_ai[i]),
                float(flops[i]),
                c=colors[i % len(colors)],
                marker=styles[1],
                linestyle="None",
                ms=markersize,
                markerfacecolor="none",
                markeredgewidth=markerwidth,
                label=labels[i] if labels else "unknown",
            )

        # In our case, on the AMD GPUs we were only able to extract metrics from the HBM
        # For adding a second value, for instance when plotting performance on the MI60 and MI100 GPU, uncomment the
        # block of code starting at line 139
        elif flag == "HBM":
            ax.plot(
                float(hbm_ai[i]),
                float(flops[i]),
                c=colors[i % len(colors)],
                marker=styles[i % len(styles)],
                linestyle="None",
                ms=markersize,
                markerfacecolor=colors[(i + 1) % len(colors)],
                markeredgewidth=markerwidth,
                label=labels[i] if labels else "unknown",
            )
            # # MI100 HBM
            # ax.plot(
            #     float(0.092024325),
            #     float(1.141301142),
            #     c=colors[6],
            #     marker=styles[8],
            #     linestyle="None",
            #     ms=markersize,
            #     markerfacecolor="none",
            #     markeredgewidth=markerwidth,
            #     label=labels[i] if labels else "unknown",
            # )

    # marker_handles = []
    #
    # if flag == "L1":
    #     marker_handles.append(
    #         ax.plot(
    #             [],
    #             [],
    #             c="k",
    #             marker=styles[0],
    #             linestyle="None",
    #             ms=markersize,
    #             markerfacecolor="none",
    #             markeredgewidth=markerwidth,
    #             label=mem_roofs[i][0],
    #         )[0]
    #     )
    # elif flag == "L2":
    #     marker_handles.append(
    #         ax.plot(
    #             [],
    #             [],
    #             c="k",
    #             marker=styles[1],
    #             linestyle="None",
    #             ms=markersize,
    #             markerfacecolor="none",
    #             markeredgewidth=markerwidth,
    #             label=mem_roofs[i][0],
    #         )[0]
    #     )
    #
    # # Uncomment lines 156 and 157 if plotting more than one device's performance
    # elif flag == "HBM":
    #     marker_handles.append(
    #         ax.plot(
    #             [],
    #             [],
    #             c=colors[3],
    #             marker=styles[2],
    #             linestyle="None",
    #             ms=markersize,
    #             markerfacecolor="none",
    #             markeredgewidth=markerwidth,
    #             label=mem_roofs[i][0],
    #         )[0]
    #     )
    #
    #     # marker_handles.append(ax.plot([], [], c=colors[6], marker=styles[8], linestyle='None', ms=markersize,
    #     #                               markerfacecolor='none', markeredgewidth=markerwidth, label=mem_roofs[0][0])[0])
    #
    cmp_roof_count = 0
    for i, roof in enumerate(cmp_roofs):
        if cmp_roof_count == 0:
            ax.text(
                x[-ixx],
                roof[1] * 1024,
                roof[0] + ": " + str(float(roof[1]) * 1000) + " wavefront GIPS",
                horizontalalignment="right",
                verticalalignment="bottom",
                color=colors[i % len(colors)],
            )
            cmp_roof_count += 1

    # If the memory bandwidth rooflines appear too close together when the plot is generated, the text indicating
    # their respective bandwidths will overlap with the rooflines. For that reason, this loop is commented out

    # for roof in mem_roofs:
    #     ang = np.arctan(np.log10(xlim[1] / xlim[0]) / np.log10(ylim[1] / ylim[0])
    #                     * fig.get_size_inches()[1] / fig.get_size_inches()[0])
    #     if x[ixx] * roof[1] > ymin:
    #         ax.text(x[ixx], x[ixx] * roof[1] * (1 + 0.25 * np.sin(ang) ** 2),
    #                 roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
    #                 horizontalalignment='left',
    #                 verticalalignment='bottom',
    #                 rotation=180 / np.pi * ang)
    #     else:
    #         ymin_ix_elbow = list()
    #         ymin_x_elbow = list()
    #         for ix in range(1, nx):
    #             if roof[1] * x[ix] >= ymin > roof[1] * x[ix - 1]:
    #                 ymin_x_elbow.append(x[ix - 1])
    #                 ymin_ix_elbow.append(ix - 1)
    #                 break
    #         ax.text(x[ixx + ymin_ix_elbow[0]], x[ixx + ymin_ix_elbow[0]] * roof[1] * (1 + 0.25 * np.sin(ang) ** 2),
    #                 roof[0] + ': ' + '{0:.1f}'.format(float(roof[1])) + ' GB/s',
    #                 horizontalalignment='left',
    #                 verticalalignment='bottom',
    #                 rotation=180 / np.pi * ang)

    # leg1 = plt.legend(
    #     handles=marker_handles,
    #     loc="lower right",
    #     ncol=len(flag[0]) if "all" not in flag else 3,
    #     bbox_to_anchor=(1, 0),
    # )
    # ax.add_artist(leg1)

    patch_handles = []
    for i in range(0, len(hbm_ai)):
        if True:
            # if flops[i] > 0.0:
            patch_handles.append(
                mpatches.Patch(
                    color=colors[i % len(colors)],
                    label=labels[i] if labels else "unknown",
                )
            )

    ax.text(
        xlim[0] * 1.1,
        ylim[1] / 1.1,
        "AMD MI100 Instruction Roofline Model",
        horizontalalignment="left",
        verticalalignment="top",
    )

    plt.savefig(f"{plot_path.stem}_roofline.png")


def calculate_roofline(profiled_path, accu_mode="mean", wavefront_size=64):
    with open(profiled_path, "r") as f:
        # Go  throw empty rows
        cnt = 0
        while True:
            ln = f.readline()
            if not ln:
                break
            cnt += 1
            if "Host Name" in ln:
                break
        df = pd.read_csv(profiled_path)
        df_sum = df.groupby("KernelName")[
            [
                "KernelName",
                "WriteSize",
                "FetchSize",
                "SQ_INSTS_VALU",
                "SQ_INSTS_SALU",
                "DurationNs",
            ]
        ].agg(accu_mode)

        df_count = df.groupby("KernelName")[["Index"]].agg({"Index": "count"})
        df_count.rename(columns={"Index": "Num Calls"}, inplace=True)
        dft = pd.merge(df_sum, df_count, on="KernelName", how="inner")
        dft["Time(s)"] = float(dft["DurationNs"]) * pow(10, -9)
        dft["Arithmetic_Instructions"] = (dft["SQ_INSTS_VALU"] * 4) + (
            dft["SQ_INSTS_SALU"]
        )
        dft["AI_HBM(FLOPs/B)"] = (
            (dft["Arithmetic_Instructions"] / wavefront_size) / dft["Time(s)"]
        ) / (((dft["FetchSize"] + dft["WriteSize"]) * 1024) / dft["Time(s)"])
        dft["GFLOP/s"] = (
            (dft["Arithmetic_Instructions"] / wavefront_size) / pow(10, 9)
        ) / dft["Time(s)"]
        print(dft)

        dft.to_csv(f"{profiled_path.stem}-roofline.csv")

        return dft


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python roofline_plot.py <data_file.csv> <plot_file.png>")
        sys.exit(1)

    data_file = Path(sys.argv[1])
    plot_file = Path(sys.argv[2])

    # calculate_roofline(data_file, accu_mode="sum")
    df = calculate_roofline(data_file, wavefront_size=64)
    # LABELS = df.index.tolist()
    # Instruction_Intensity_HBM = df["AI_HBM(FLOPs/B)"].tolist()
    # GFLOPs = df["GFLOP/s"].tolist()
    #
    # print(LABELS)
    # print(Instruction_Intensity_HBM)
    # print(GFLOPs)
    #
    # roofline(plot_file, GFLOPs, Instruction_Intensity_HBM, [0], [0], LABELS)
