#!/usr/bin/env perl

use strict;
use warnings;
local $SIG{__WARN__} = sub { die $_[0] };

use List::Util qw(max);
use Getopt::Long qw(:config no_ignore_case);

chomp(my $file = `readlink -f $0`);
my $directory = $1 if($file =~ s/(^.*\/)//);

GetOptions(
	'h' => \(my $help = ''),
);
if($help || scalar(@ARGV) == 0) {
	die <<EOF;

Usage:   $file [options] table.txt > table.transposed.txt

Options: -h       display this help message

EOF
}
my ($tableFile) = @ARGV;
my @tokenListList = ();
open(my $reader, $tableFile);
while(my $line = <$reader>) {
	chomp($line);
	my @tokenList = split(/\t/, $line, -1);
	push(@tokenList, '') if(scalar(@tokenList) == 0);
	push(@tokenListList, \@tokenList);
}
close($reader);
foreach my $index (0 .. max(map {$#$_} @tokenListList)) {
	print join("\t", map {defined($_) ? $_ : ''} map {$_->[$index]} @tokenListList), "\n";
}
