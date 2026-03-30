# rmsf_backbone.tcl
#
# Run with:
#   vmd -dispdev text -e rmsf_backbone.tcl
#
# Edit these file names/types for your AMBER simulation:

set topfile   "md.prmtop"
set trajfiles  [list md_step7a.nc md_step8a.nc]
set trajtype  "netcdf"   ;# use "netcdf", "mdcrd", or another VMD-supported type
set outfile   "MD_step7-8A_backbone_rmsf.dat"

# -----------------------------
# Load topology and trajectory
# -----------------------------
mol new $topfile type parm7 waitfor all

foreach traj $trajfiles {
    mol addfile $traj type $trajtype waitfor all
}

set molid top
set nframes [molinfo $molid get numframes]

if {$nframes < 2} {
    puts "WARNING: Trajectory has only $nframes frame(s)."
}

# ---------------------------------------------------------
# Align every frame to frame 0 using protein backbone atoms
# ---------------------------------------------------------
set refsel [atomselect $molid "protein and backbone" frame 0]

for {set i 0} {$i < $nframes} {incr i} {
    set mobsel [atomselect $molid "protein and backbone" frame $i]
    set allsel [atomselect $molid "all" frame $i]

    set trans_mat [measure fit $mobsel $refsel]
    $allsel move $trans_mat

    $mobsel delete
    $allsel delete
}

$refsel delete

# ---------------------------------------------------------
# Compute RMSF for all protein backbone atoms over trajectory
# ---------------------------------------------------------
set bbsel [atomselect $molid "protein and backbone"]
set atom_rmsf [measure rmsf $bbsel first 0 last [expr {$nframes - 1}] step 1]

# Get residue IDs corresponding to each backbone atom
set resid_list [$bbsel get resid]

# ---------------------------------------------------------
# Average backbone-atom RMSF values per residue
# ---------------------------------------------------------
array set rmsf_sum {}
array set rmsf_count {}

set natoms [llength $resid_list]
for {set i 0} {$i < $natoms} {incr i} {
    set resid [lindex $resid_list $i]
    set rmsf  [lindex $atom_rmsf $i]

    if {![info exists rmsf_sum($resid)]} {
        set rmsf_sum($resid) 0.0
        set rmsf_count($resid) 0
    }

    set rmsf_sum($resid) [expr {$rmsf_sum($resid) + $rmsf}]
    incr rmsf_count($resid)
}

# Sort residue IDs numerically
set sorted_resids [lsort -integer [array names rmsf_sum]]

# -----------------------------
# Write output file
# -----------------------------
set fh [open $outfile w]
puts $fh "# resid  avg_backbone_RMSF_A"

foreach resid $sorted_resids {
    set avg_rmsf [expr {$rmsf_sum($resid) / $rmsf_count($resid)}]
    puts $fh [format "%d %.6f" $resid $avg_rmsf]
}

close $fh
$bbsel delete

puts "Done."
puts "Wrote per-residue backbone RMSF to: $outfile"

quit
