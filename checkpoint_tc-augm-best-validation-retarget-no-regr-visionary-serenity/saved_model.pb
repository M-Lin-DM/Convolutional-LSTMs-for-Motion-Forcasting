??4
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
convLSTM_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconvLSTM_1/kernel

%convLSTM_1/kernel/Read/ReadVariableOpReadVariableOpconvLSTM_1/kernel*&
_output_shapes
:@*
dtype0
?
convLSTM_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameconvLSTM_1/recurrent_kernel
?
/convLSTM_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpconvLSTM_1/recurrent_kernel*&
_output_shapes
:@*
dtype0
?
time_distributed_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name time_distributed_1/moving_mean
?
2time_distributed_1/moving_mean/Read/ReadVariableOpReadVariableOptime_distributed_1/moving_mean*
_output_shapes
:*
dtype0
?
"time_distributed_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"time_distributed_1/moving_variance
?
6time_distributed_1/moving_variance/Read/ReadVariableOpReadVariableOp"time_distributed_1/moving_variance*
_output_shapes
:*
dtype0
?
convLSTM_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameconvLSTM_2/kernel
?
%convLSTM_2/kernel/Read/ReadVariableOpReadVariableOpconvLSTM_2/kernel*'
_output_shapes
:?*
dtype0
?
convLSTM_2/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: ?*,
shared_nameconvLSTM_2/recurrent_kernel
?
/convLSTM_2/recurrent_kernel/Read/ReadVariableOpReadVariableOpconvLSTM_2/recurrent_kernel*'
_output_shapes
: ?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:?*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:?*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:?*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/convLSTM_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/convLSTM_1/kernel/m
?
,Adam/convLSTM_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/convLSTM_1/kernel/m*&
_output_shapes
:@*
dtype0
?
"Adam/convLSTM_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/convLSTM_1/recurrent_kernel/m
?
6Adam/convLSTM_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp"Adam/convLSTM_1/recurrent_kernel/m*&
_output_shapes
:@*
dtype0
?
Adam/convLSTM_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/convLSTM_2/kernel/m
?
,Adam/convLSTM_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/convLSTM_2/kernel/m*'
_output_shapes
:?*
dtype0
?
"Adam/convLSTM_2/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: ?*3
shared_name$"Adam/convLSTM_2/recurrent_kernel/m
?
6Adam/convLSTM_2/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp"Adam/convLSTM_2/recurrent_kernel/m*'
_output_shapes
: ?*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/convLSTM_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/convLSTM_1/kernel/v
?
,Adam/convLSTM_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/convLSTM_1/kernel/v*&
_output_shapes
:@*
dtype0
?
"Adam/convLSTM_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/convLSTM_1/recurrent_kernel/v
?
6Adam/convLSTM_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp"Adam/convLSTM_1/recurrent_kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/convLSTM_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_nameAdam/convLSTM_2/kernel/v
?
,Adam/convLSTM_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/convLSTM_2/kernel/v*'
_output_shapes
:?*
dtype0
?
"Adam/convLSTM_2/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: ?*3
shared_name$"Adam/convLSTM_2/recurrent_kernel/v
?
6Adam/convLSTM_2/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp"Adam/convLSTM_2/recurrent_kernel/v*'
_output_shapes
: ?*
dtype0
?
ConstConst*
_output_shapes
:*
dtype0*U
valueLBJ"@  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??  ??
?
Const_1Const*
_output_shapes
:*
dtype0*U
valueLBJ"@                                                                

NoOpNoOp
?=
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?<
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
 
l
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
]
	layer
	variables
regularization_losses
trainable_variables
	keras_api
]
	layer
	variables
regularization_losses
trainable_variables
	keras_api
l
cell

state_spec
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?
.iter

/beta_1

0beta_2
	1decay
2learning_rate(m?)m?3m?4m?7m?8m?(v?)v?3v?4v?7v?8v?
8
30
41
52
63
74
85
(6
)7
 
*
30
41
72
83
(4
)5
?
		variables
9layer_metrics

regularization_losses
:metrics
;non_trainable_variables

<layers
=layer_regularization_losses
trainable_variables
 
t

3kernel
4recurrent_kernel
>	variables
?regularization_losses
@trainable_variables
A	keras_api
 

30
41
 

30
41
?

Bstates
	variables
Clayer_metrics
regularization_losses
Dmetrics
Enon_trainable_variables

Flayers
Glayer_regularization_losses
trainable_variables
R
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
 
 
 
?
	variables
Llayer_metrics
regularization_losses
Mmetrics
Nnon_trainable_variables

Olayers
Player_regularization_losses
trainable_variables
?
Qaxis
5moving_mean
6moving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api

50
61
 
 
?
	variables
Vlayer_metrics
regularization_losses
Wmetrics
Xnon_trainable_variables

Ylayers
Zlayer_regularization_losses
trainable_variables
t

7kernel
8recurrent_kernel
[	variables
\regularization_losses
]trainable_variables
^	keras_api
 

70
81
 

70
81
?

_states
 	variables
`layer_metrics
!regularization_losses
ametrics
bnon_trainable_variables

clayers
dlayer_regularization_losses
"trainable_variables
 
 
 
?
$	variables
elayer_metrics
%regularization_losses
fmetrics
gnon_trainable_variables

hlayers
ilayer_regularization_losses
&trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
*	variables
jlayer_metrics
+regularization_losses
kmetrics
lnon_trainable_variables

mlayers
nlayer_regularization_losses
,trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconvLSTM_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLSTM_1/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEtime_distributed_1/moving_mean&variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"time_distributed_1/moving_variance&variables/3/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconvLSTM_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconvLSTM_2/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1
q2

50
61
1
0
1
2
3
4
5
6
 

30
41
 

30
41
?
>	variables
rlayer_metrics
?regularization_losses
smetrics
tnon_trainable_variables

ulayers
vlayer_regularization_losses
@trainable_variables
 
 
 
 

0
 
 
 
 
?
H	variables
wlayer_metrics
Iregularization_losses
xmetrics
ynon_trainable_variables

zlayers
{layer_regularization_losses
Jtrainable_variables
 
 
 

0
 
 

50
61
 
 
?
R	variables
|layer_metrics
Sregularization_losses
}metrics
~non_trainable_variables

layers
 ?layer_regularization_losses
Ttrainable_variables
 
 

50
61

0
 

70
81
 

70
81
?
[	variables
?layer_metrics
\regularization_losses
?metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
]trainable_variables
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
v
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 

50
61
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/convLSTM_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/convLSTM_1/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/convLSTM_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/convLSTM_2/recurrent_kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/convLSTM_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/convLSTM_1/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/convLSTM_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/convLSTM_2/recurrent_kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*4
_output_shapes"
 :?????????x?*
dtype0*)
shape :?????????x?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1convLSTM_1/kernelconvLSTM_1/recurrent_kernelConstConst_1time_distributed_1/moving_mean"time_distributed_1/moving_varianceconvLSTM_2/kernelconvLSTM_2/recurrent_kerneldense/kernel
dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_604985
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp%convLSTM_1/kernel/Read/ReadVariableOp/convLSTM_1/recurrent_kernel/Read/ReadVariableOp2time_distributed_1/moving_mean/Read/ReadVariableOp6time_distributed_1/moving_variance/Read/ReadVariableOp%convLSTM_2/kernel/Read/ReadVariableOp/convLSTM_2/recurrent_kernel/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp,Adam/convLSTM_1/kernel/m/Read/ReadVariableOp6Adam/convLSTM_1/recurrent_kernel/m/Read/ReadVariableOp,Adam/convLSTM_2/kernel/m/Read/ReadVariableOp6Adam/convLSTM_2/recurrent_kernel/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp,Adam/convLSTM_1/kernel/v/Read/ReadVariableOp6Adam/convLSTM_1/recurrent_kernel/v/Read/ReadVariableOp,Adam/convLSTM_2/kernel/v/Read/ReadVariableOp6Adam/convLSTM_2/recurrent_kernel/v/Read/ReadVariableOpConst_2*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_608427
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconvLSTM_1/kernelconvLSTM_1/recurrent_kerneltime_distributed_1/moving_mean"time_distributed_1/moving_varianceconvLSTM_2/kernelconvLSTM_2/recurrent_kerneltotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativesAdam/dense/kernel/mAdam/dense/bias/mAdam/convLSTM_1/kernel/m"Adam/convLSTM_1/recurrent_kernel/mAdam/convLSTM_2/kernel/m"Adam/convLSTM_2/recurrent_kernel/mAdam/dense/kernel/vAdam/dense/bias/vAdam/convLSTM_1/kernel/v"Adam/convLSTM_1/recurrent_kernel/vAdam/convLSTM_2/kernel/v"Adam/convLSTM_2/recurrent_kernel/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_608536??/
?
?
+__inference_convLSTM_2_layer_call_fn_607823

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6047342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????;O::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?g
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_604734

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_8}
addAddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_604618*
condR
while_cond_604617*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const|
IdentityIdentitystrided_slice_2:output:0^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????;O::2
whilewhile:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?
?
while_cond_607264
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_607264___redundant_placeholder04
0while_while_cond_607264___redundant_placeholder14
0while_while_cond_607264___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?	
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608107

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????;O2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????;O:::::W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_1_layer_call_fn_606906

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????;O*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6031762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:&??????????????????;O::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_1_layer_call_fn_606974

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6042912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????;O::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?
?
3__inference_time_distributed_1_layer_call_fn_606893

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6031372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:&??????????????????;O::::22
StatefulPartitionedCallStatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?
?
+__inference_convLSTM_1_layer_call_fn_606737

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6039942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????v?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????x?::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?6
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_602796

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?whileu

zeros_like	ZerosLikeinputs*
T0*=
_output_shapes+
):'??????????????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'??????????????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:?????????v?:?????????v?:?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_6023792
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_602731*
condR
while_cond_602730*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'??????????????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'??????????????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*
T0*=
_output_shapes+
):'??????????????????v?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'??????????????????x?::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:e a
=
_output_shapes+
):'??????????????????x?
 
_user_specified_nameinputs
?
?
while_cond_603877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_603877___redundant_placeholder04
0while_while_cond_603877___redundant_placeholder14
0while_while_cond_603877___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_602730
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_602730___redundant_placeholder04
0while_while_cond_602730___redundant_placeholder14
0while_while_cond_602730___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?g
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607178
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilev

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&??????????????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_8}
addAddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_607062*
condR
while_cond_607061*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const|
IdentityIdentitystrided_slice_2:output:0^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&??????????????????;O::2
whilewhile:f b
<
_output_shapes*
(:&??????????????????;O
"
_user_specified_name
inputs/0
?8
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_603770

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?whilet

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&??????????????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????9M :?????????9M :?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_6033472
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_603705*
condR
while_cond_603704*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_2:output:0^StatefulPartitionedCall^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&??????????????????;O::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?"
?
while_body_603705
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_603729_0
while_603731_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_603729
while_603731??while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_603729_0while_603731_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????9M :?????????9M :?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_6033472
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1~
while/IdentityIdentitywhile/add_1:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5"
while_603729while_603729_0"
while_603731while_603731_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?X
?
while_body_603878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:?????????v?2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_602955

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608091

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????;O2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????;O::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs
?'
?
H__inference_functional_1_layer_call_and_return_conditional_losses_604925

inputs
convlstm_1_604892
convlstm_1_604894
time_distributed_1_604900
time_distributed_1_604902
time_distributed_1_604904
time_distributed_1_604906
convlstm_2_604911
convlstm_2_604913
dense_604917
dense_604919
identity??"convLSTM_1/StatefulPartitionedCall?"convLSTM_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinputsconvlstm_1_604892convlstm_1_604894*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6041952$
"convLSTM_1/StatefulPartitionedCall?
 time_distributed/PartitionedCallPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6042362"
 time_distributed/PartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape+convLSTM_1/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0time_distributed_1_604900time_distributed_1_604902time_distributed_1_604904time_distributed_1_604906*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6042912,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape)time_distributed/PartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape?
"convLSTM_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0convlstm_2_604911convlstm_2_604913*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6047342$
"convLSTM_2/StatefulPartitionedCall?
$global_max_pooling2d/PartitionedCallPartitionedCall+convLSTM_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_6037842&
$global_max_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_604917dense_604919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6047702
dense/StatefulPartitionedCall?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^convLSTM_1/StatefulPartitionedCall#^convLSTM_2/StatefulPartitionedCall^dense/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2H
"convLSTM_2/StatefulPartitionedCall"convLSTM_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?X
?
while_body_606411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:?????????v?2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?
M
1__inference_time_distributed_layer_call_fn_606815

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6028732
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'??????????????????v?:e a
=
_output_shapes+
):'??????????????????v?
 
_user_specified_nameinputs
?'
?
H__inference_functional_1_layer_call_and_return_conditional_losses_604825
input_1
convlstm_1_604792
convlstm_1_604794
time_distributed_1_604800
time_distributed_1_604802
time_distributed_1_604804
time_distributed_1_604806
convlstm_2_604811
convlstm_2_604813
dense_604817
dense_604819
identity??"convLSTM_1/StatefulPartitionedCall?"convLSTM_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinput_1convlstm_1_604792convlstm_1_604794*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6041952$
"convLSTM_1/StatefulPartitionedCall?
 time_distributed/PartitionedCallPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6042362"
 time_distributed/PartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape+convLSTM_1/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0time_distributed_1_604800time_distributed_1_604802time_distributed_1_604804time_distributed_1_604806*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6042912,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape)time_distributed/PartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape?
"convLSTM_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0convlstm_2_604811convlstm_2_604813*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6047342$
"convLSTM_2/StatefulPartitionedCall?
$global_max_pooling2d/PartitionedCallPartitionedCall+convLSTM_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_6037842&
$global_max_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_604817dense_604819*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6047702
dense/StatefulPartitionedCall?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^convLSTM_1/StatefulPartitionedCall#^convLSTM_2/StatefulPartitionedCall^dense/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2H
"convLSTM_2/StatefulPartitionedCall"convLSTM_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????x?
!
_user_specified_name	input_1
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_604236

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????v?:\ X
4
_output_shapes"
 :?????????v?
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_603176

inputs
batchnorm_1_603160
batchnorm_1_603162
batchnorm_1_603164
batchnorm_1_603166
identity??#batchnorm_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0batchnorm_1_603160batchnorm_1_603162batchnorm_1_603164batchnorm_1_603166*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????;O*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_6030392%
#batchnorm_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape,batchnorm_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1?
IdentityIdentityReshape_1:output:0$^batchnorm_1/StatefulPartitionedCall*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:&??????????????????;O::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?;
?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_602379

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constf
IdentityIdentity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:?????????x?:?????????v?:?????????v?:::X T
0
_output_shapes
:?????????x?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????v?
 
_user_specified_namestates:XT
0
_output_shapes
:?????????v?
 
_user_specified_namestates
?	
?
-__inference_functional_1_layer_call_fn_605880

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6048642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?;
?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_607978

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constf
IdentityIdentity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:?????????x?:?????????v?:?????????v?:::X T
0
_output_shapes
:?????????x?
 
_user_specified_nameinputs:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/0:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/1
?n
?
convLSTM_1_while_body_6055062
.convlstm_1_while_convlstm_1_while_loop_counter8
4convlstm_1_while_convlstm_1_while_maximum_iterations 
convlstm_1_while_placeholder"
convlstm_1_while_placeholder_1"
convlstm_1_while_placeholder_2"
convlstm_1_while_placeholder_3/
+convlstm_1_while_convlstm_1_strided_slice_0m
iconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_04
0convlstm_1_while_split_readvariableop_resource_06
2convlstm_1_while_split_1_readvariableop_resource_0
convlstm_1_while_identity
convlstm_1_while_identity_1
convlstm_1_while_identity_2
convlstm_1_while_identity_3
convlstm_1_while_identity_4
convlstm_1_while_identity_5-
)convlstm_1_while_convlstm_1_strided_slicek
gconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor2
.convlstm_1_while_split_readvariableop_resource4
0convlstm_1_while_split_1_readvariableop_resource??
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      2D
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0convlstm_1_while_placeholderKconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype026
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemr
convLSTM_1/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/Const?
 convLSTM_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/while/split/split_dim?
%convLSTM_1/while/split/ReadVariableOpReadVariableOp0convlstm_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02'
%convLSTM_1/while/split/ReadVariableOp?
convLSTM_1/while/splitSplit)convLSTM_1/while/split/split_dim:output:0-convLSTM_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/while/splitv
convLSTM_1/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/Const_1?
"convLSTM_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"convLSTM_1/while/split_1/split_dim?
'convLSTM_1/while/split_1/ReadVariableOpReadVariableOp2convlstm_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02)
'convLSTM_1/while/split_1/ReadVariableOp?
convLSTM_1/while/split_1Split+convLSTM_1/while/split_1/split_dim:output:0/convLSTM_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/while/split_1?
convLSTM_1/while/convolutionConv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/while/convolution?
convLSTM_1/while/convolution_1Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_1?
convLSTM_1/while/convolution_2Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_2?
convLSTM_1/while/convolution_3Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_3?
convLSTM_1/while/convolution_4Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_4?
convLSTM_1/while/convolution_5Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_5?
convLSTM_1/while/convolution_6Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_6?
convLSTM_1/while/convolution_7Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_7?
convLSTM_1/while/addAddV2%convLSTM_1/while/convolution:output:0'convLSTM_1/while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/addy
convLSTM_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/while/Const_2y
convLSTM_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_3?
convLSTM_1/while/MulMulconvLSTM_1/while/add:z:0!convLSTM_1/while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Mul?
convLSTM_1/while/Add_1AddconvLSTM_1/while/Mul:z:0!convLSTM_1/while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Add_1?
(convLSTM_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(convLSTM_1/while/clip_by_value/Minimum/y?
&convLSTM_1/while/clip_by_value/MinimumMinimumconvLSTM_1/while/Add_1:z:01convLSTM_1/while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2(
&convLSTM_1/while/clip_by_value/Minimum?
 convLSTM_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 convLSTM_1/while/clip_by_value/y?
convLSTM_1/while/clip_by_valueMaximum*convLSTM_1/while/clip_by_value/Minimum:z:0)convLSTM_1/while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2 
convLSTM_1/while/clip_by_value?
convLSTM_1/while/add_2AddV2'convLSTM_1/while/convolution_1:output:0'convLSTM_1/while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_2y
convLSTM_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/while/Const_4y
convLSTM_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_5?
convLSTM_1/while/Mul_1MulconvLSTM_1/while/add_2:z:0!convLSTM_1/while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Mul_1?
convLSTM_1/while/Add_3AddconvLSTM_1/while/Mul_1:z:0!convLSTM_1/while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Add_3?
*convLSTM_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_1/while/clip_by_value_1/Minimum/y?
(convLSTM_1/while/clip_by_value_1/MinimumMinimumconvLSTM_1/while/Add_3:z:03convLSTM_1/while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2*
(convLSTM_1/while/clip_by_value_1/Minimum?
"convLSTM_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_1/y?
 convLSTM_1/while/clip_by_value_1Maximum,convLSTM_1/while/clip_by_value_1/Minimum:z:0+convLSTM_1/while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2"
 convLSTM_1/while/clip_by_value_1?
convLSTM_1/while/mul_2Mul$convLSTM_1/while/clip_by_value_1:z:0convlstm_1_while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/mul_2?
convLSTM_1/while/add_4AddV2'convLSTM_1/while/convolution_2:output:0'convLSTM_1/while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_4?
convLSTM_1/while/TanhTanhconvLSTM_1/while/add_4:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Tanh?
convLSTM_1/while/mul_3Mul"convLSTM_1/while/clip_by_value:z:0convLSTM_1/while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/mul_3?
convLSTM_1/while/add_5AddV2convLSTM_1/while/mul_2:z:0convLSTM_1/while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_5?
convLSTM_1/while/add_6AddV2'convLSTM_1/while/convolution_3:output:0'convLSTM_1/while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_6y
convLSTM_1/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/while/Const_6y
convLSTM_1/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_7?
convLSTM_1/while/Mul_4MulconvLSTM_1/while/add_6:z:0!convLSTM_1/while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Mul_4?
convLSTM_1/while/Add_7AddconvLSTM_1/while/Mul_4:z:0!convLSTM_1/while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Add_7?
*convLSTM_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_1/while/clip_by_value_2/Minimum/y?
(convLSTM_1/while/clip_by_value_2/MinimumMinimumconvLSTM_1/while/Add_7:z:03convLSTM_1/while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2*
(convLSTM_1/while/clip_by_value_2/Minimum?
"convLSTM_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_2/y?
 convLSTM_1/while/clip_by_value_2Maximum,convLSTM_1/while/clip_by_value_2/Minimum:z:0+convLSTM_1/while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2"
 convLSTM_1/while/clip_by_value_2?
convLSTM_1/while/Tanh_1TanhconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Tanh_1?
convLSTM_1/while/mul_5Mul$convLSTM_1/while/clip_by_value_2:z:0convLSTM_1/while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/mul_5?
5convLSTM_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemconvlstm_1_while_placeholder_1convlstm_1_while_placeholderconvLSTM_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype027
5convLSTM_1/while/TensorArrayV2Write/TensorListSetItemv
convLSTM_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/add_8/y?
convLSTM_1/while/add_8AddV2convlstm_1_while_placeholder!convLSTM_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2
convLSTM_1/while/add_8v
convLSTM_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/add_9/y?
convLSTM_1/while/add_9AddV2.convlstm_1_while_convlstm_1_while_loop_counter!convLSTM_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2
convLSTM_1/while/add_9
convLSTM_1/while/IdentityIdentityconvLSTM_1/while/add_9:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity?
convLSTM_1/while/Identity_1Identity4convlstm_1_while_convlstm_1_while_maximum_iterations*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_1?
convLSTM_1/while/Identity_2IdentityconvLSTM_1/while/add_8:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_2?
convLSTM_1/while/Identity_3IdentityEconvLSTM_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_3?
convLSTM_1/while/Identity_4IdentityconvLSTM_1/while/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Identity_4?
convLSTM_1/while/Identity_5IdentityconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Identity_5"X
)convlstm_1_while_convlstm_1_strided_slice+convlstm_1_while_convlstm_1_strided_slice_0"?
convlstm_1_while_identity"convLSTM_1/while/Identity:output:0"C
convlstm_1_while_identity_1$convLSTM_1/while/Identity_1:output:0"C
convlstm_1_while_identity_2$convLSTM_1/while/Identity_2:output:0"C
convlstm_1_while_identity_3$convLSTM_1/while/Identity_3:output:0"C
convlstm_1_while_identity_4$convLSTM_1/while/Identity_4:output:0"C
convlstm_1_while_identity_5$convLSTM_1/while/Identity_5:output:0"f
0convlstm_1_while_split_1_readvariableop_resource2convlstm_1_while_split_1_readvariableop_resource_0"b
.convlstm_1_while_split_readvariableop_resource0convlstm_1_while_split_readvariableop_resource_0"?
gconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensoriconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?"
?
while_body_602731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_602755_0
while_602757_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_602755
while_602757??while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_602755_0while_602757_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:?????????v?:?????????v?:?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_6023792
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1~
while/IdentityIdentitywhile/add_1:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2
while/Identity_5"
while_602755while_602755_0"
while_602757while_602757_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_607485
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_607485___redundant_placeholder04
0while_while_cond_607485___redundant_placeholder14
0while_while_cond_607485___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?g
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607602

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_8}
addAddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_607486*
condR
while_cond_607485*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const|
IdentityIdentitystrided_slice_2:output:0^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????;O::2
whilewhile:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?
?
4__inference_conv_lst_m2d_cell_1_layer_call_fn_608283

inputs
states_0
states_1
unknown
	unknown_0
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????9M :?????????9M :?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_6032802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*l
_input_shapes[
Y:?????????;O:?????????9M :?????????9M ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/1
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_606764

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????v?:\ X
4
_output_shapes"
 :?????????v?
 
_user_specified_nameinputs
?
?
)functional_1_convLSTM_1_while_cond_601873L
Hfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_loop_counterR
Nfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_maximum_iterations-
)functional_1_convlstm_1_while_placeholder/
+functional_1_convlstm_1_while_placeholder_1/
+functional_1_convlstm_1_while_placeholder_2/
+functional_1_convlstm_1_while_placeholder_3L
Hfunctional_1_convlstm_1_while_less_functional_1_convlstm_1_strided_sliced
`functional_1_convlstm_1_while_functional_1_convlstm_1_while_cond_601873___redundant_placeholder0d
`functional_1_convlstm_1_while_functional_1_convlstm_1_while_cond_601873___redundant_placeholder1d
`functional_1_convlstm_1_while_functional_1_convlstm_1_while_cond_601873___redundant_placeholder2*
&functional_1_convlstm_1_while_identity
?
"functional_1/convLSTM_1/while/LessLess)functional_1_convlstm_1_while_placeholderHfunctional_1_convlstm_1_while_less_functional_1_convlstm_1_strided_slice*
T0*
_output_shapes
: 2$
"functional_1/convLSTM_1/while/Less?
&functional_1/convLSTM_1/while/IdentityIdentity&functional_1/convLSTM_1/while/Less:z:0*
T0
*
_output_shapes
: 2(
&functional_1/convLSTM_1/while/Identity"Y
&functional_1_convlstm_1_while_identity/functional_1/convLSTM_1/while/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?X
?
while_body_604415
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3~
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_4k

while/TanhTanhwhile/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2o
while/Tanh_1Tanhwhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identitywhile/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4{
while/Identity_5Identitywhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_606792

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'??????????????????v?:e a
=
_output_shapes+
):'??????????????????v?
 
_user_specified_nameinputs
?
M
1__inference_time_distributed_layer_call_fn_606774

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6042362
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????v?:\ X
4
_output_shapes"
 :?????????v?
 
_user_specified_nameinputs
?X
?
while_body_606192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:?????????v?2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?n
?
convLSTM_2_while_body_6057292
.convlstm_2_while_convlstm_2_while_loop_counter8
4convlstm_2_while_convlstm_2_while_maximum_iterations 
convlstm_2_while_placeholder"
convlstm_2_while_placeholder_1"
convlstm_2_while_placeholder_2"
convlstm_2_while_placeholder_3/
+convlstm_2_while_convlstm_2_strided_slice_0m
iconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor_04
0convlstm_2_while_split_readvariableop_resource_06
2convlstm_2_while_split_1_readvariableop_resource_0
convlstm_2_while_identity
convlstm_2_while_identity_1
convlstm_2_while_identity_2
convlstm_2_while_identity_3
convlstm_2_while_identity_4
convlstm_2_while_identity_5-
)convlstm_2_while_convlstm_2_strided_slicek
gconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor2
.convlstm_2_while_split_readvariableop_resource4
0convlstm_2_while_split_1_readvariableop_resource??
BconvLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2D
BconvLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4convLSTM_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0convlstm_2_while_placeholderKconvLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype026
4convLSTM_2/while/TensorArrayV2Read/TensorListGetItemr
convLSTM_2/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/Const?
 convLSTM_2/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_2/while/split/split_dim?
%convLSTM_2/while/split/ReadVariableOpReadVariableOp0convlstm_2_while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02'
%convLSTM_2/while/split/ReadVariableOp?
convLSTM_2/while/splitSplit)convLSTM_2/while/split/split_dim:output:0-convLSTM_2/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
convLSTM_2/while/splitv
convLSTM_2/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/Const_1?
"convLSTM_2/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"convLSTM_2/while/split_1/split_dim?
'convLSTM_2/while/split_1/ReadVariableOpReadVariableOp2convlstm_2_while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02)
'convLSTM_2/while/split_1/ReadVariableOp?
convLSTM_2/while/split_1Split+convLSTM_2/while/split_1/split_dim:output:0/convLSTM_2/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
convLSTM_2/while/split_1?
convLSTM_2/while/convolutionConv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/while/convolution?
convLSTM_2/while/convolution_1Conv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2 
convLSTM_2/while/convolution_1?
convLSTM_2/while/convolution_2Conv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2 
convLSTM_2/while/convolution_2?
convLSTM_2/while/convolution_3Conv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2 
convLSTM_2/while/convolution_3?
convLSTM_2/while/convolution_4Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_4?
convLSTM_2/while/convolution_5Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_5?
convLSTM_2/while/convolution_6Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_6?
convLSTM_2/while/convolution_7Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_7?
convLSTM_2/while/addAddV2%convLSTM_2/while/convolution:output:0'convLSTM_2/while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/addy
convLSTM_2/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/while/Const_2y
convLSTM_2/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/while/Const_3?
convLSTM_2/while/MulMulconvLSTM_2/while/add:z:0!convLSTM_2/while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Mul?
convLSTM_2/while/Add_1AddconvLSTM_2/while/Mul:z:0!convLSTM_2/while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Add_1?
(convLSTM_2/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(convLSTM_2/while/clip_by_value/Minimum/y?
&convLSTM_2/while/clip_by_value/MinimumMinimumconvLSTM_2/while/Add_1:z:01convLSTM_2/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2(
&convLSTM_2/while/clip_by_value/Minimum?
 convLSTM_2/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 convLSTM_2/while/clip_by_value/y?
convLSTM_2/while/clip_by_valueMaximum*convLSTM_2/while/clip_by_value/Minimum:z:0)convLSTM_2/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2 
convLSTM_2/while/clip_by_value?
convLSTM_2/while/add_2AddV2'convLSTM_2/while/convolution_1:output:0'convLSTM_2/while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_2y
convLSTM_2/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/while/Const_4y
convLSTM_2/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/while/Const_5?
convLSTM_2/while/Mul_1MulconvLSTM_2/while/add_2:z:0!convLSTM_2/while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Mul_1?
convLSTM_2/while/Add_3AddconvLSTM_2/while/Mul_1:z:0!convLSTM_2/while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Add_3?
*convLSTM_2/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_2/while/clip_by_value_1/Minimum/y?
(convLSTM_2/while/clip_by_value_1/MinimumMinimumconvLSTM_2/while/Add_3:z:03convLSTM_2/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2*
(convLSTM_2/while/clip_by_value_1/Minimum?
"convLSTM_2/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_2/while/clip_by_value_1/y?
 convLSTM_2/while/clip_by_value_1Maximum,convLSTM_2/while/clip_by_value_1/Minimum:z:0+convLSTM_2/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2"
 convLSTM_2/while/clip_by_value_1?
convLSTM_2/while/mul_2Mul$convLSTM_2/while/clip_by_value_1:z:0convlstm_2_while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/mul_2?
convLSTM_2/while/add_4AddV2'convLSTM_2/while/convolution_2:output:0'convLSTM_2/while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_4?
convLSTM_2/while/TanhTanhconvLSTM_2/while/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Tanh?
convLSTM_2/while/mul_3Mul"convLSTM_2/while/clip_by_value:z:0convLSTM_2/while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/mul_3?
convLSTM_2/while/add_5AddV2convLSTM_2/while/mul_2:z:0convLSTM_2/while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_5?
convLSTM_2/while/add_6AddV2'convLSTM_2/while/convolution_3:output:0'convLSTM_2/while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_6y
convLSTM_2/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/while/Const_6y
convLSTM_2/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/while/Const_7?
convLSTM_2/while/Mul_4MulconvLSTM_2/while/add_6:z:0!convLSTM_2/while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Mul_4?
convLSTM_2/while/Add_7AddconvLSTM_2/while/Mul_4:z:0!convLSTM_2/while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Add_7?
*convLSTM_2/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_2/while/clip_by_value_2/Minimum/y?
(convLSTM_2/while/clip_by_value_2/MinimumMinimumconvLSTM_2/while/Add_7:z:03convLSTM_2/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2*
(convLSTM_2/while/clip_by_value_2/Minimum?
"convLSTM_2/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_2/while/clip_by_value_2/y?
 convLSTM_2/while/clip_by_value_2Maximum,convLSTM_2/while/clip_by_value_2/Minimum:z:0+convLSTM_2/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2"
 convLSTM_2/while/clip_by_value_2?
convLSTM_2/while/Tanh_1TanhconvLSTM_2/while/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Tanh_1?
convLSTM_2/while/mul_5Mul$convLSTM_2/while/clip_by_value_2:z:0convLSTM_2/while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/mul_5?
5convLSTM_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemconvlstm_2_while_placeholder_1convlstm_2_while_placeholderconvLSTM_2/while/mul_5:z:0*
_output_shapes
: *
element_dtype027
5convLSTM_2/while/TensorArrayV2Write/TensorListSetItemv
convLSTM_2/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/add_8/y?
convLSTM_2/while/add_8AddV2convlstm_2_while_placeholder!convLSTM_2/while/add_8/y:output:0*
T0*
_output_shapes
: 2
convLSTM_2/while/add_8v
convLSTM_2/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/add_9/y?
convLSTM_2/while/add_9AddV2.convlstm_2_while_convlstm_2_while_loop_counter!convLSTM_2/while/add_9/y:output:0*
T0*
_output_shapes
: 2
convLSTM_2/while/add_9
convLSTM_2/while/IdentityIdentityconvLSTM_2/while/add_9:z:0*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity?
convLSTM_2/while/Identity_1Identity4convlstm_2_while_convlstm_2_while_maximum_iterations*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity_1?
convLSTM_2/while/Identity_2IdentityconvLSTM_2/while/add_8:z:0*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity_2?
convLSTM_2/while/Identity_3IdentityEconvLSTM_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity_3?
convLSTM_2/while/Identity_4IdentityconvLSTM_2/while/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Identity_4?
convLSTM_2/while/Identity_5IdentityconvLSTM_2/while/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Identity_5"X
)convlstm_2_while_convlstm_2_strided_slice+convlstm_2_while_convlstm_2_strided_slice_0"?
convlstm_2_while_identity"convLSTM_2/while/Identity:output:0"C
convlstm_2_while_identity_1$convLSTM_2/while/Identity_1:output:0"C
convlstm_2_while_identity_2$convLSTM_2/while/Identity_2:output:0"C
convlstm_2_while_identity_3$convLSTM_2/while/Identity_3:output:0"C
convlstm_2_while_identity_4$convLSTM_2/while/Identity_4:output:0"C
convlstm_2_while_identity_5$convLSTM_2/while/Identity_5:output:0"f
0convlstm_2_while_split_1_readvariableop_resource2convlstm_2_while_split_1_readvariableop_resource_0"b
.convlstm_2_while_split_readvariableop_resource0convlstm_2_while_split_readvariableop_resource_0"?
gconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensoriconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?n
?
convLSTM_1_while_body_6050702
.convlstm_1_while_convlstm_1_while_loop_counter8
4convlstm_1_while_convlstm_1_while_maximum_iterations 
convlstm_1_while_placeholder"
convlstm_1_while_placeholder_1"
convlstm_1_while_placeholder_2"
convlstm_1_while_placeholder_3/
+convlstm_1_while_convlstm_1_strided_slice_0m
iconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_04
0convlstm_1_while_split_readvariableop_resource_06
2convlstm_1_while_split_1_readvariableop_resource_0
convlstm_1_while_identity
convlstm_1_while_identity_1
convlstm_1_while_identity_2
convlstm_1_while_identity_3
convlstm_1_while_identity_4
convlstm_1_while_identity_5-
)convlstm_1_while_convlstm_1_strided_slicek
gconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor2
.convlstm_1_while_split_readvariableop_resource4
0convlstm_1_while_split_1_readvariableop_resource??
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      2D
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0convlstm_1_while_placeholderKconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype026
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemr
convLSTM_1/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/Const?
 convLSTM_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/while/split/split_dim?
%convLSTM_1/while/split/ReadVariableOpReadVariableOp0convlstm_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02'
%convLSTM_1/while/split/ReadVariableOp?
convLSTM_1/while/splitSplit)convLSTM_1/while/split/split_dim:output:0-convLSTM_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/while/splitv
convLSTM_1/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/Const_1?
"convLSTM_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"convLSTM_1/while/split_1/split_dim?
'convLSTM_1/while/split_1/ReadVariableOpReadVariableOp2convlstm_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02)
'convLSTM_1/while/split_1/ReadVariableOp?
convLSTM_1/while/split_1Split+convLSTM_1/while/split_1/split_dim:output:0/convLSTM_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/while/split_1?
convLSTM_1/while/convolutionConv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/while/convolution?
convLSTM_1/while/convolution_1Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_1?
convLSTM_1/while/convolution_2Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_2?
convLSTM_1/while/convolution_3Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_3?
convLSTM_1/while/convolution_4Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_4?
convLSTM_1/while/convolution_5Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_5?
convLSTM_1/while/convolution_6Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_6?
convLSTM_1/while/convolution_7Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_7?
convLSTM_1/while/addAddV2%convLSTM_1/while/convolution:output:0'convLSTM_1/while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/addy
convLSTM_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/while/Const_2y
convLSTM_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_3?
convLSTM_1/while/MulMulconvLSTM_1/while/add:z:0!convLSTM_1/while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Mul?
convLSTM_1/while/Add_1AddconvLSTM_1/while/Mul:z:0!convLSTM_1/while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Add_1?
(convLSTM_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(convLSTM_1/while/clip_by_value/Minimum/y?
&convLSTM_1/while/clip_by_value/MinimumMinimumconvLSTM_1/while/Add_1:z:01convLSTM_1/while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2(
&convLSTM_1/while/clip_by_value/Minimum?
 convLSTM_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 convLSTM_1/while/clip_by_value/y?
convLSTM_1/while/clip_by_valueMaximum*convLSTM_1/while/clip_by_value/Minimum:z:0)convLSTM_1/while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2 
convLSTM_1/while/clip_by_value?
convLSTM_1/while/add_2AddV2'convLSTM_1/while/convolution_1:output:0'convLSTM_1/while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_2y
convLSTM_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/while/Const_4y
convLSTM_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_5?
convLSTM_1/while/Mul_1MulconvLSTM_1/while/add_2:z:0!convLSTM_1/while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Mul_1?
convLSTM_1/while/Add_3AddconvLSTM_1/while/Mul_1:z:0!convLSTM_1/while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Add_3?
*convLSTM_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_1/while/clip_by_value_1/Minimum/y?
(convLSTM_1/while/clip_by_value_1/MinimumMinimumconvLSTM_1/while/Add_3:z:03convLSTM_1/while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2*
(convLSTM_1/while/clip_by_value_1/Minimum?
"convLSTM_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_1/y?
 convLSTM_1/while/clip_by_value_1Maximum,convLSTM_1/while/clip_by_value_1/Minimum:z:0+convLSTM_1/while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2"
 convLSTM_1/while/clip_by_value_1?
convLSTM_1/while/mul_2Mul$convLSTM_1/while/clip_by_value_1:z:0convlstm_1_while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/mul_2?
convLSTM_1/while/add_4AddV2'convLSTM_1/while/convolution_2:output:0'convLSTM_1/while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_4?
convLSTM_1/while/TanhTanhconvLSTM_1/while/add_4:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Tanh?
convLSTM_1/while/mul_3Mul"convLSTM_1/while/clip_by_value:z:0convLSTM_1/while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/mul_3?
convLSTM_1/while/add_5AddV2convLSTM_1/while/mul_2:z:0convLSTM_1/while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_5?
convLSTM_1/while/add_6AddV2'convLSTM_1/while/convolution_3:output:0'convLSTM_1/while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/add_6y
convLSTM_1/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/while/Const_6y
convLSTM_1/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_7?
convLSTM_1/while/Mul_4MulconvLSTM_1/while/add_6:z:0!convLSTM_1/while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Mul_4?
convLSTM_1/while/Add_7AddconvLSTM_1/while/Mul_4:z:0!convLSTM_1/while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Add_7?
*convLSTM_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_1/while/clip_by_value_2/Minimum/y?
(convLSTM_1/while/clip_by_value_2/MinimumMinimumconvLSTM_1/while/Add_7:z:03convLSTM_1/while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2*
(convLSTM_1/while/clip_by_value_2/Minimum?
"convLSTM_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_2/y?
 convLSTM_1/while/clip_by_value_2Maximum,convLSTM_1/while/clip_by_value_2/Minimum:z:0+convLSTM_1/while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2"
 convLSTM_1/while/clip_by_value_2?
convLSTM_1/while/Tanh_1TanhconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Tanh_1?
convLSTM_1/while/mul_5Mul$convLSTM_1/while/clip_by_value_2:z:0convLSTM_1/while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/mul_5?
5convLSTM_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemconvlstm_1_while_placeholder_1convlstm_1_while_placeholderconvLSTM_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype027
5convLSTM_1/while/TensorArrayV2Write/TensorListSetItemv
convLSTM_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/add_8/y?
convLSTM_1/while/add_8AddV2convlstm_1_while_placeholder!convLSTM_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2
convLSTM_1/while/add_8v
convLSTM_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/add_9/y?
convLSTM_1/while/add_9AddV2.convlstm_1_while_convlstm_1_while_loop_counter!convLSTM_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2
convLSTM_1/while/add_9
convLSTM_1/while/IdentityIdentityconvLSTM_1/while/add_9:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity?
convLSTM_1/while/Identity_1Identity4convlstm_1_while_convlstm_1_while_maximum_iterations*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_1?
convLSTM_1/while/Identity_2IdentityconvLSTM_1/while/add_8:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_2?
convLSTM_1/while/Identity_3IdentityEconvLSTM_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_3?
convLSTM_1/while/Identity_4IdentityconvLSTM_1/while/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Identity_4?
convLSTM_1/while/Identity_5IdentityconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/while/Identity_5"X
)convlstm_1_while_convlstm_1_strided_slice+convlstm_1_while_convlstm_1_strided_slice_0"?
convlstm_1_while_identity"convLSTM_1/while/Identity:output:0"C
convlstm_1_while_identity_1$convLSTM_1/while/Identity_1:output:0"C
convlstm_1_while_identity_2$convLSTM_1/while/Identity_2:output:0"C
convlstm_1_while_identity_3$convLSTM_1/while/Identity_3:output:0"C
convlstm_1_while_identity_4$convLSTM_1/while/Identity_4:output:0"C
convlstm_1_while_identity_5$convLSTM_1/while/Identity_5:output:0"f
0convlstm_1_while_split_1_readvariableop_resource2convlstm_1_while_split_1_readvariableop_resource_0"b
.convlstm_1_while_split_readvariableop_resource0convlstm_1_while_split_readvariableop_resource_0"?
gconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensoriconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_603137

inputs
batchnorm_1_603121
batchnorm_1_603123
batchnorm_1_603125
batchnorm_1_603127
identity??#batchnorm_1/StatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0batchnorm_1_603121batchnorm_1_603123batchnorm_1_603125batchnorm_1_603127*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_6030232%
#batchnorm_1/StatefulPartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape,batchnorm_1/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1?
IdentityIdentityReshape_1:output:0$^batchnorm_1/StatefulPartitionedCall*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:&??????????????????;O::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606880

inputs
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3Reshape:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 2
batchnorm_1/FusedBatchNormV3q
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape batchnorm_1/FusedBatchNormV3:y:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:&??????????????????;O:::::d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?"
?
while_body_602623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_602647_0
while_602649_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_602647
while_602649??while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_602647_0while_602649_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:?????????v?:?????????v?:?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_6023122
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1~
while/IdentityIdentitywhile/add_1:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2
while/Identity_5"
while_602647while_602647_0"
while_602649while_602649_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_607688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_607688___redundant_placeholder04
0while_while_cond_607688___redundant_placeholder14
0while_while_cond_607688___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?6
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_602688

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?whileu

zeros_like	ZerosLikeinputs*
T0*=
_output_shapes+
):'??????????????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'??????????????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:?????????v?:?????????v?:?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_6023122
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_602623*
condR
while_cond_602622*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'??????????????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'??????????????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
IdentityIdentitytranspose_1:y:0^StatefulPartitionedCall^while*
T0*=
_output_shapes+
):'??????????????????v?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'??????????????????x?::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:e a
=
_output_shapes+
):'??????????????????x?
 
_user_specified_nameinputs
?
?
+__inference_convLSTM_1_layer_call_fn_606746

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6041952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????v?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????x?::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?
?
+__inference_convLSTM_2_layer_call_fn_607814

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6045312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????;O::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606948

inputs
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource
identity?w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3Reshape:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 2
batchnorm_1/FusedBatchNormV3
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshape batchnorm_1/FusedBatchNormV3:y:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????;O:::::[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?;
?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_603280

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7{
addAddV2convolution:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Conste
IdentityIdentity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identityi

Identity_1Identity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_1i

Identity_2Identity	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*l
_input_shapes[
Y:?????????;O:?????????9M :?????????9M :::W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????9M 
 
_user_specified_namestates:WS
/
_output_shapes
:?????????9M 
 
_user_specified_namestates
??
?
!__inference__wrapped_model_602221
input_19
5functional_1_convlstm_1_split_readvariableop_resource;
7functional_1_convlstm_1_split_1_readvariableop_resource5
1functional_1_time_distributed_1_batchnorm_1_scale6
2functional_1_time_distributed_1_batchnorm_1_offsetX
Tfunctional_1_time_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resourceZ
Vfunctional_1_time_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource9
5functional_1_convlstm_2_split_readvariableop_resource;
7functional_1_convlstm_2_split_1_readvariableop_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource
identity??functional_1/convLSTM_1/while?functional_1/convLSTM_2/while?
"functional_1/convLSTM_1/zeros_like	ZerosLikeinput_1*
T0*4
_output_shapes"
 :?????????x?2$
"functional_1/convLSTM_1/zeros_like?
-functional_1/convLSTM_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/convLSTM_1/Sum/reduction_indices?
functional_1/convLSTM_1/SumSum&functional_1/convLSTM_1/zeros_like:y:06functional_1/convLSTM_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
functional_1/convLSTM_1/Sum?
functional_1/convLSTM_1/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
functional_1/convLSTM_1/zeros?
#functional_1/convLSTM_1/convolutionConv2D$functional_1/convLSTM_1/Sum:output:0&functional_1/convLSTM_1/zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2%
#functional_1/convLSTM_1/convolution?
&functional_1/convLSTM_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2(
&functional_1/convLSTM_1/transpose/perm?
!functional_1/convLSTM_1/transpose	Transposeinput_1/functional_1/convLSTM_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2#
!functional_1/convLSTM_1/transpose?
functional_1/convLSTM_1/ShapeShape%functional_1/convLSTM_1/transpose:y:0*
T0*
_output_shapes
:2
functional_1/convLSTM_1/Shape?
+functional_1/convLSTM_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/convLSTM_1/strided_slice/stack?
-functional_1/convLSTM_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/convLSTM_1/strided_slice/stack_1?
-functional_1/convLSTM_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/convLSTM_1/strided_slice/stack_2?
%functional_1/convLSTM_1/strided_sliceStridedSlice&functional_1/convLSTM_1/Shape:output:04functional_1/convLSTM_1/strided_slice/stack:output:06functional_1/convLSTM_1/strided_slice/stack_1:output:06functional_1/convLSTM_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/convLSTM_1/strided_slice?
3functional_1/convLSTM_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3functional_1/convLSTM_1/TensorArrayV2/element_shape?
%functional_1/convLSTM_1/TensorArrayV2TensorListReserve<functional_1/convLSTM_1/TensorArrayV2/element_shape:output:0.functional_1/convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%functional_1/convLSTM_1/TensorArrayV2?
Mfunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      2O
Mfunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
?functional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%functional_1/convLSTM_1/transpose:y:0Vfunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?functional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor?
-functional_1/convLSTM_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/convLSTM_1/strided_slice_1/stack?
/functional_1/convLSTM_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_1/strided_slice_1/stack_1?
/functional_1/convLSTM_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_1/strided_slice_1/stack_2?
'functional_1/convLSTM_1/strided_slice_1StridedSlice%functional_1/convLSTM_1/transpose:y:06functional_1/convLSTM_1/strided_slice_1/stack:output:08functional_1/convLSTM_1/strided_slice_1/stack_1:output:08functional_1/convLSTM_1/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2)
'functional_1/convLSTM_1/strided_slice_1?
functional_1/convLSTM_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
functional_1/convLSTM_1/Const?
'functional_1/convLSTM_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'functional_1/convLSTM_1/split/split_dim?
,functional_1/convLSTM_1/split/ReadVariableOpReadVariableOp5functional_1_convlstm_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,functional_1/convLSTM_1/split/ReadVariableOp?
functional_1/convLSTM_1/splitSplit0functional_1/convLSTM_1/split/split_dim:output:04functional_1/convLSTM_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
functional_1/convLSTM_1/split?
functional_1/convLSTM_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/convLSTM_1/Const_1?
)functional_1/convLSTM_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)functional_1/convLSTM_1/split_1/split_dim?
.functional_1/convLSTM_1/split_1/ReadVariableOpReadVariableOp7functional_1_convlstm_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype020
.functional_1/convLSTM_1/split_1/ReadVariableOp?
functional_1/convLSTM_1/split_1Split2functional_1/convLSTM_1/split_1/split_dim:output:06functional_1/convLSTM_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2!
functional_1/convLSTM_1/split_1?
%functional_1/convLSTM_1/convolution_1Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_1?
%functional_1/convLSTM_1/convolution_2Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_2?
%functional_1/convLSTM_1/convolution_3Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_3?
%functional_1/convLSTM_1/convolution_4Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_4?
%functional_1/convLSTM_1/convolution_5Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_5?
%functional_1/convLSTM_1/convolution_6Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_6?
%functional_1/convLSTM_1/convolution_7Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_7?
%functional_1/convLSTM_1/convolution_8Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_8?
functional_1/convLSTM_1/addAddV2.functional_1/convLSTM_1/convolution_1:output:0.functional_1/convLSTM_1/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/add?
functional_1/convLSTM_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
functional_1/convLSTM_1/Const_2?
functional_1/convLSTM_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_1/Const_3?
functional_1/convLSTM_1/MulMulfunctional_1/convLSTM_1/add:z:0(functional_1/convLSTM_1/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Mul?
functional_1/convLSTM_1/Add_1Addfunctional_1/convLSTM_1/Mul:z:0(functional_1/convLSTM_1/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Add_1?
/functional_1/convLSTM_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??21
/functional_1/convLSTM_1/clip_by_value/Minimum/y?
-functional_1/convLSTM_1/clip_by_value/MinimumMinimum!functional_1/convLSTM_1/Add_1:z:08functional_1/convLSTM_1/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2/
-functional_1/convLSTM_1/clip_by_value/Minimum?
'functional_1/convLSTM_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'functional_1/convLSTM_1/clip_by_value/y?
%functional_1/convLSTM_1/clip_by_valueMaximum1functional_1/convLSTM_1/clip_by_value/Minimum:z:00functional_1/convLSTM_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2'
%functional_1/convLSTM_1/clip_by_value?
functional_1/convLSTM_1/add_2AddV2.functional_1/convLSTM_1/convolution_2:output:0.functional_1/convLSTM_1/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/add_2?
functional_1/convLSTM_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
functional_1/convLSTM_1/Const_4?
functional_1/convLSTM_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_1/Const_5?
functional_1/convLSTM_1/Mul_1Mul!functional_1/convLSTM_1/add_2:z:0(functional_1/convLSTM_1/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Mul_1?
functional_1/convLSTM_1/Add_3Add!functional_1/convLSTM_1/Mul_1:z:0(functional_1/convLSTM_1/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Add_3?
1functional_1/convLSTM_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1functional_1/convLSTM_1/clip_by_value_1/Minimum/y?
/functional_1/convLSTM_1/clip_by_value_1/MinimumMinimum!functional_1/convLSTM_1/Add_3:z:0:functional_1/convLSTM_1/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?21
/functional_1/convLSTM_1/clip_by_value_1/Minimum?
)functional_1/convLSTM_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)functional_1/convLSTM_1/clip_by_value_1/y?
'functional_1/convLSTM_1/clip_by_value_1Maximum3functional_1/convLSTM_1/clip_by_value_1/Minimum:z:02functional_1/convLSTM_1/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2)
'functional_1/convLSTM_1/clip_by_value_1?
functional_1/convLSTM_1/mul_2Mul+functional_1/convLSTM_1/clip_by_value_1:z:0,functional_1/convLSTM_1/convolution:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/mul_2?
functional_1/convLSTM_1/add_4AddV2.functional_1/convLSTM_1/convolution_3:output:0.functional_1/convLSTM_1/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/add_4?
functional_1/convLSTM_1/TanhTanh!functional_1/convLSTM_1/add_4:z:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Tanh?
functional_1/convLSTM_1/mul_3Mul)functional_1/convLSTM_1/clip_by_value:z:0 functional_1/convLSTM_1/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/mul_3?
functional_1/convLSTM_1/add_5AddV2!functional_1/convLSTM_1/mul_2:z:0!functional_1/convLSTM_1/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/add_5?
functional_1/convLSTM_1/add_6AddV2.functional_1/convLSTM_1/convolution_4:output:0.functional_1/convLSTM_1/convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/add_6?
functional_1/convLSTM_1/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
functional_1/convLSTM_1/Const_6?
functional_1/convLSTM_1/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_1/Const_7?
functional_1/convLSTM_1/Mul_4Mul!functional_1/convLSTM_1/add_6:z:0(functional_1/convLSTM_1/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Mul_4?
functional_1/convLSTM_1/Add_7Add!functional_1/convLSTM_1/Mul_4:z:0(functional_1/convLSTM_1/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/Add_7?
1functional_1/convLSTM_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1functional_1/convLSTM_1/clip_by_value_2/Minimum/y?
/functional_1/convLSTM_1/clip_by_value_2/MinimumMinimum!functional_1/convLSTM_1/Add_7:z:0:functional_1/convLSTM_1/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?21
/functional_1/convLSTM_1/clip_by_value_2/Minimum?
)functional_1/convLSTM_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)functional_1/convLSTM_1/clip_by_value_2/y?
'functional_1/convLSTM_1/clip_by_value_2Maximum3functional_1/convLSTM_1/clip_by_value_2/Minimum:z:02functional_1/convLSTM_1/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2)
'functional_1/convLSTM_1/clip_by_value_2?
functional_1/convLSTM_1/Tanh_1Tanh!functional_1/convLSTM_1/add_5:z:0*
T0*0
_output_shapes
:?????????v?2 
functional_1/convLSTM_1/Tanh_1?
functional_1/convLSTM_1/mul_5Mul+functional_1/convLSTM_1/clip_by_value_2:z:0"functional_1/convLSTM_1/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
functional_1/convLSTM_1/mul_5?
5functional_1/convLSTM_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      27
5functional_1/convLSTM_1/TensorArrayV2_1/element_shape?
'functional_1/convLSTM_1/TensorArrayV2_1TensorListReserve>functional_1/convLSTM_1/TensorArrayV2_1/element_shape:output:0.functional_1/convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'functional_1/convLSTM_1/TensorArrayV2_1~
functional_1/convLSTM_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
functional_1/convLSTM_1/time?
0functional_1/convLSTM_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0functional_1/convLSTM_1/while/maximum_iterations?
*functional_1/convLSTM_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/convLSTM_1/while/loop_counter?
functional_1/convLSTM_1/whileWhile3functional_1/convLSTM_1/while/loop_counter:output:09functional_1/convLSTM_1/while/maximum_iterations:output:0%functional_1/convLSTM_1/time:output:00functional_1/convLSTM_1/TensorArrayV2_1:handle:0,functional_1/convLSTM_1/convolution:output:0,functional_1/convLSTM_1/convolution:output:0.functional_1/convLSTM_1/strided_slice:output:0Ofunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor:output_handle:05functional_1_convlstm_1_split_readvariableop_resource7functional_1_convlstm_1_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*5
body-R+
)functional_1_convLSTM_1_while_body_601874*5
cond-R+
)functional_1_convLSTM_1_while_cond_601873*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
functional_1/convLSTM_1/while?
Hfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2J
Hfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape?
:functional_1/convLSTM_1/TensorArrayV2Stack/TensorListStackTensorListStack&functional_1/convLSTM_1/while:output:3Qfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02<
:functional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack?
-functional_1/convLSTM_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-functional_1/convLSTM_1/strided_slice_2/stack?
/functional_1/convLSTM_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/convLSTM_1/strided_slice_2/stack_1?
/functional_1/convLSTM_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_1/strided_slice_2/stack_2?
'functional_1/convLSTM_1/strided_slice_2StridedSliceCfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:06functional_1/convLSTM_1/strided_slice_2/stack:output:08functional_1/convLSTM_1/strided_slice_2/stack_1:output:08functional_1/convLSTM_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2)
'functional_1/convLSTM_1/strided_slice_2?
(functional_1/convLSTM_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2*
(functional_1/convLSTM_1/transpose_1/perm?
#functional_1/convLSTM_1/transpose_1	TransposeCfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:01functional_1/convLSTM_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2%
#functional_1/convLSTM_1/transpose_1?
+functional_1/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2-
+functional_1/time_distributed/Reshape/shape?
%functional_1/time_distributed/ReshapeReshape'functional_1/convLSTM_1/transpose_1:y:04functional_1/time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2'
%functional_1/time_distributed/Reshape?
3functional_1/time_distributed/max_pooling2d/MaxPoolMaxPool.functional_1/time_distributed/Reshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
25
3functional_1/time_distributed/max_pooling2d/MaxPool?
-functional_1/time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2/
-functional_1/time_distributed/Reshape_1/shape?
'functional_1/time_distributed/Reshape_1Reshape<functional_1/time_distributed/max_pooling2d/MaxPool:output:06functional_1/time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2)
'functional_1/time_distributed/Reshape_1?
-functional_1/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2/
-functional_1/time_distributed/Reshape_2/shape?
'functional_1/time_distributed/Reshape_2Reshape'functional_1/convLSTM_1/transpose_1:y:06functional_1/time_distributed/Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????v?2)
'functional_1/time_distributed/Reshape_2?
-functional_1/time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2/
-functional_1/time_distributed_1/Reshape/shape?
'functional_1/time_distributed_1/ReshapeReshape0functional_1/time_distributed/Reshape_1:output:06functional_1/time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2)
'functional_1/time_distributed_1/Reshape?
Kfunctional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOpTfunctional_1_time_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02M
Kfunctional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp?
Mfunctional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVfunctional_1_time_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02O
Mfunctional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
<functional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3FusedBatchNormV30functional_1/time_distributed_1/Reshape:output:01functional_1_time_distributed_1_batchnorm_1_scale2functional_1_time_distributed_1_batchnorm_1_offsetSfunctional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0Ufunctional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 2>
<functional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3?
/functional_1/time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      21
/functional_1/time_distributed_1/Reshape_1/shape?
)functional_1/time_distributed_1/Reshape_1Reshape@functional_1/time_distributed_1/batchnorm_1/FusedBatchNormV3:y:08functional_1/time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2+
)functional_1/time_distributed_1/Reshape_1?
/functional_1/time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      21
/functional_1/time_distributed_1/Reshape_2/shape?
)functional_1/time_distributed_1/Reshape_2Reshape0functional_1/time_distributed/Reshape_1:output:08functional_1/time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????;O2+
)functional_1/time_distributed_1/Reshape_2?
"functional_1/convLSTM_2/zeros_like	ZerosLike2functional_1/time_distributed_1/Reshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2$
"functional_1/convLSTM_2/zeros_like?
-functional_1/convLSTM_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/convLSTM_2/Sum/reduction_indices?
functional_1/convLSTM_2/SumSum&functional_1/convLSTM_2/zeros_like:y:06functional_1/convLSTM_2/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
functional_1/convLSTM_2/Sum?
-functional_1/convLSTM_2/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-functional_1/convLSTM_2/zeros/shape_as_tensor?
#functional_1/convLSTM_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#functional_1/convLSTM_2/zeros/Const?
functional_1/convLSTM_2/zerosFill6functional_1/convLSTM_2/zeros/shape_as_tensor:output:0,functional_1/convLSTM_2/zeros/Const:output:0*
T0*&
_output_shapes
: 2
functional_1/convLSTM_2/zeros?
#functional_1/convLSTM_2/convolutionConv2D$functional_1/convLSTM_2/Sum:output:0&functional_1/convLSTM_2/zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2%
#functional_1/convLSTM_2/convolution?
&functional_1/convLSTM_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2(
&functional_1/convLSTM_2/transpose/perm?
!functional_1/convLSTM_2/transpose	Transpose2functional_1/time_distributed_1/Reshape_1:output:0/functional_1/convLSTM_2/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2#
!functional_1/convLSTM_2/transpose?
functional_1/convLSTM_2/ShapeShape%functional_1/convLSTM_2/transpose:y:0*
T0*
_output_shapes
:2
functional_1/convLSTM_2/Shape?
+functional_1/convLSTM_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/convLSTM_2/strided_slice/stack?
-functional_1/convLSTM_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/convLSTM_2/strided_slice/stack_1?
-functional_1/convLSTM_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/convLSTM_2/strided_slice/stack_2?
%functional_1/convLSTM_2/strided_sliceStridedSlice&functional_1/convLSTM_2/Shape:output:04functional_1/convLSTM_2/strided_slice/stack:output:06functional_1/convLSTM_2/strided_slice/stack_1:output:06functional_1/convLSTM_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/convLSTM_2/strided_slice?
3functional_1/convLSTM_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????25
3functional_1/convLSTM_2/TensorArrayV2/element_shape?
%functional_1/convLSTM_2/TensorArrayV2TensorListReserve<functional_1/convLSTM_2/TensorArrayV2/element_shape:output:0.functional_1/convLSTM_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%functional_1/convLSTM_2/TensorArrayV2?
Mfunctional_1/convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2O
Mfunctional_1/convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
?functional_1/convLSTM_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%functional_1/convLSTM_2/transpose:y:0Vfunctional_1/convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?functional_1/convLSTM_2/TensorArrayUnstack/TensorListFromTensor?
-functional_1/convLSTM_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/convLSTM_2/strided_slice_1/stack?
/functional_1/convLSTM_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_2/strided_slice_1/stack_1?
/functional_1/convLSTM_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_2/strided_slice_1/stack_2?
'functional_1/convLSTM_2/strided_slice_1StridedSlice%functional_1/convLSTM_2/transpose:y:06functional_1/convLSTM_2/strided_slice_1/stack:output:08functional_1/convLSTM_2/strided_slice_1/stack_1:output:08functional_1/convLSTM_2/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2)
'functional_1/convLSTM_2/strided_slice_1?
functional_1/convLSTM_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
functional_1/convLSTM_2/Const?
'functional_1/convLSTM_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'functional_1/convLSTM_2/split/split_dim?
,functional_1/convLSTM_2/split/ReadVariableOpReadVariableOp5functional_1_convlstm_2_split_readvariableop_resource*'
_output_shapes
:?*
dtype02.
,functional_1/convLSTM_2/split/ReadVariableOp?
functional_1/convLSTM_2/splitSplit0functional_1/convLSTM_2/split/split_dim:output:04functional_1/convLSTM_2/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
functional_1/convLSTM_2/split?
functional_1/convLSTM_2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/convLSTM_2/Const_1?
)functional_1/convLSTM_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)functional_1/convLSTM_2/split_1/split_dim?
.functional_1/convLSTM_2/split_1/ReadVariableOpReadVariableOp7functional_1_convlstm_2_split_1_readvariableop_resource*'
_output_shapes
: ?*
dtype020
.functional_1/convLSTM_2/split_1/ReadVariableOp?
functional_1/convLSTM_2/split_1Split2functional_1/convLSTM_2/split_1/split_dim:output:06functional_1/convLSTM_2/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2!
functional_1/convLSTM_2/split_1?
%functional_1/convLSTM_2/convolution_1Conv2D0functional_1/convLSTM_2/strided_slice_1:output:0&functional_1/convLSTM_2/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2'
%functional_1/convLSTM_2/convolution_1?
%functional_1/convLSTM_2/convolution_2Conv2D0functional_1/convLSTM_2/strided_slice_1:output:0&functional_1/convLSTM_2/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2'
%functional_1/convLSTM_2/convolution_2?
%functional_1/convLSTM_2/convolution_3Conv2D0functional_1/convLSTM_2/strided_slice_1:output:0&functional_1/convLSTM_2/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2'
%functional_1/convLSTM_2/convolution_3?
%functional_1/convLSTM_2/convolution_4Conv2D0functional_1/convLSTM_2/strided_slice_1:output:0&functional_1/convLSTM_2/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2'
%functional_1/convLSTM_2/convolution_4?
%functional_1/convLSTM_2/convolution_5Conv2D,functional_1/convLSTM_2/convolution:output:0(functional_1/convLSTM_2/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2'
%functional_1/convLSTM_2/convolution_5?
%functional_1/convLSTM_2/convolution_6Conv2D,functional_1/convLSTM_2/convolution:output:0(functional_1/convLSTM_2/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2'
%functional_1/convLSTM_2/convolution_6?
%functional_1/convLSTM_2/convolution_7Conv2D,functional_1/convLSTM_2/convolution:output:0(functional_1/convLSTM_2/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2'
%functional_1/convLSTM_2/convolution_7?
%functional_1/convLSTM_2/convolution_8Conv2D,functional_1/convLSTM_2/convolution:output:0(functional_1/convLSTM_2/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2'
%functional_1/convLSTM_2/convolution_8?
functional_1/convLSTM_2/addAddV2.functional_1/convLSTM_2/convolution_1:output:0.functional_1/convLSTM_2/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/add?
functional_1/convLSTM_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
functional_1/convLSTM_2/Const_2?
functional_1/convLSTM_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_2/Const_3?
functional_1/convLSTM_2/MulMulfunctional_1/convLSTM_2/add:z:0(functional_1/convLSTM_2/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Mul?
functional_1/convLSTM_2/Add_1Addfunctional_1/convLSTM_2/Mul:z:0(functional_1/convLSTM_2/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Add_1?
/functional_1/convLSTM_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??21
/functional_1/convLSTM_2/clip_by_value/Minimum/y?
-functional_1/convLSTM_2/clip_by_value/MinimumMinimum!functional_1/convLSTM_2/Add_1:z:08functional_1/convLSTM_2/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2/
-functional_1/convLSTM_2/clip_by_value/Minimum?
'functional_1/convLSTM_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'functional_1/convLSTM_2/clip_by_value/y?
%functional_1/convLSTM_2/clip_by_valueMaximum1functional_1/convLSTM_2/clip_by_value/Minimum:z:00functional_1/convLSTM_2/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2'
%functional_1/convLSTM_2/clip_by_value?
functional_1/convLSTM_2/add_2AddV2.functional_1/convLSTM_2/convolution_2:output:0.functional_1/convLSTM_2/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/add_2?
functional_1/convLSTM_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
functional_1/convLSTM_2/Const_4?
functional_1/convLSTM_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_2/Const_5?
functional_1/convLSTM_2/Mul_1Mul!functional_1/convLSTM_2/add_2:z:0(functional_1/convLSTM_2/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Mul_1?
functional_1/convLSTM_2/Add_3Add!functional_1/convLSTM_2/Mul_1:z:0(functional_1/convLSTM_2/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Add_3?
1functional_1/convLSTM_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1functional_1/convLSTM_2/clip_by_value_1/Minimum/y?
/functional_1/convLSTM_2/clip_by_value_1/MinimumMinimum!functional_1/convLSTM_2/Add_3:z:0:functional_1/convLSTM_2/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 21
/functional_1/convLSTM_2/clip_by_value_1/Minimum?
)functional_1/convLSTM_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)functional_1/convLSTM_2/clip_by_value_1/y?
'functional_1/convLSTM_2/clip_by_value_1Maximum3functional_1/convLSTM_2/clip_by_value_1/Minimum:z:02functional_1/convLSTM_2/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2)
'functional_1/convLSTM_2/clip_by_value_1?
functional_1/convLSTM_2/mul_2Mul+functional_1/convLSTM_2/clip_by_value_1:z:0,functional_1/convLSTM_2/convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/mul_2?
functional_1/convLSTM_2/add_4AddV2.functional_1/convLSTM_2/convolution_3:output:0.functional_1/convLSTM_2/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/add_4?
functional_1/convLSTM_2/TanhTanh!functional_1/convLSTM_2/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Tanh?
functional_1/convLSTM_2/mul_3Mul)functional_1/convLSTM_2/clip_by_value:z:0 functional_1/convLSTM_2/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/mul_3?
functional_1/convLSTM_2/add_5AddV2!functional_1/convLSTM_2/mul_2:z:0!functional_1/convLSTM_2/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/add_5?
functional_1/convLSTM_2/add_6AddV2.functional_1/convLSTM_2/convolution_4:output:0.functional_1/convLSTM_2/convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/add_6?
functional_1/convLSTM_2/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2!
functional_1/convLSTM_2/Const_6?
functional_1/convLSTM_2/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_2/Const_7?
functional_1/convLSTM_2/Mul_4Mul!functional_1/convLSTM_2/add_6:z:0(functional_1/convLSTM_2/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Mul_4?
functional_1/convLSTM_2/Add_7Add!functional_1/convLSTM_2/Mul_4:z:0(functional_1/convLSTM_2/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/Add_7?
1functional_1/convLSTM_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1functional_1/convLSTM_2/clip_by_value_2/Minimum/y?
/functional_1/convLSTM_2/clip_by_value_2/MinimumMinimum!functional_1/convLSTM_2/Add_7:z:0:functional_1/convLSTM_2/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 21
/functional_1/convLSTM_2/clip_by_value_2/Minimum?
)functional_1/convLSTM_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)functional_1/convLSTM_2/clip_by_value_2/y?
'functional_1/convLSTM_2/clip_by_value_2Maximum3functional_1/convLSTM_2/clip_by_value_2/Minimum:z:02functional_1/convLSTM_2/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2)
'functional_1/convLSTM_2/clip_by_value_2?
functional_1/convLSTM_2/Tanh_1Tanh!functional_1/convLSTM_2/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2 
functional_1/convLSTM_2/Tanh_1?
functional_1/convLSTM_2/mul_5Mul+functional_1/convLSTM_2/clip_by_value_2:z:0"functional_1/convLSTM_2/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
functional_1/convLSTM_2/mul_5?
5functional_1/convLSTM_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       27
5functional_1/convLSTM_2/TensorArrayV2_1/element_shape?
'functional_1/convLSTM_2/TensorArrayV2_1TensorListReserve>functional_1/convLSTM_2/TensorArrayV2_1/element_shape:output:0.functional_1/convLSTM_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'functional_1/convLSTM_2/TensorArrayV2_1~
functional_1/convLSTM_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
functional_1/convLSTM_2/time?
0functional_1/convLSTM_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0functional_1/convLSTM_2/while/maximum_iterations?
*functional_1/convLSTM_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/convLSTM_2/while/loop_counter?
functional_1/convLSTM_2/whileWhile3functional_1/convLSTM_2/while/loop_counter:output:09functional_1/convLSTM_2/while/maximum_iterations:output:0%functional_1/convLSTM_2/time:output:00functional_1/convLSTM_2/TensorArrayV2_1:handle:0,functional_1/convLSTM_2/convolution:output:0,functional_1/convLSTM_2/convolution:output:0.functional_1/convLSTM_2/strided_slice:output:0Ofunctional_1/convLSTM_2/TensorArrayUnstack/TensorListFromTensor:output_handle:05functional_1_convlstm_2_split_readvariableop_resource7functional_1_convlstm_2_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*5
body-R+
)functional_1_convLSTM_2_while_body_602097*5
cond-R+
)functional_1_convLSTM_2_while_cond_602096*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
functional_1/convLSTM_2/while?
Hfunctional_1/convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2J
Hfunctional_1/convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shape?
:functional_1/convLSTM_2/TensorArrayV2Stack/TensorListStackTensorListStack&functional_1/convLSTM_2/while:output:3Qfunctional_1/convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02<
:functional_1/convLSTM_2/TensorArrayV2Stack/TensorListStack?
-functional_1/convLSTM_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2/
-functional_1/convLSTM_2/strided_slice_2/stack?
/functional_1/convLSTM_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/convLSTM_2/strided_slice_2/stack_1?
/functional_1/convLSTM_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_2/strided_slice_2/stack_2?
'functional_1/convLSTM_2/strided_slice_2StridedSliceCfunctional_1/convLSTM_2/TensorArrayV2Stack/TensorListStack:tensor:06functional_1/convLSTM_2/strided_slice_2/stack:output:08functional_1/convLSTM_2/strided_slice_2/stack_1:output:08functional_1/convLSTM_2/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2)
'functional_1/convLSTM_2/strided_slice_2?
(functional_1/convLSTM_2/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2*
(functional_1/convLSTM_2/transpose_1/perm?
#functional_1/convLSTM_2/transpose_1	TransposeCfunctional_1/convLSTM_2/TensorArrayV2Stack/TensorListStack:tensor:01functional_1/convLSTM_2/transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2%
#functional_1/convLSTM_2/transpose_1?
7functional_1/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/global_max_pooling2d/Max/reduction_indices?
%functional_1/global_max_pooling2d/MaxMax0functional_1/convLSTM_2/strided_slice_2:output:0@functional_1/global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2'
%functional_1/global_max_pooling2d/Max?
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02*
(functional_1/dense/MatMul/ReadVariableOp?
functional_1/dense/MatMulMatMul.functional_1/global_max_pooling2d/Max:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense/MatMul?
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOp?
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
functional_1/dense/BiasAdd?
functional_1/dense/SigmoidSigmoid#functional_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
functional_1/dense/Sigmoid?
IdentityIdentityfunctional_1/dense/Sigmoid:y:0^functional_1/convLSTM_1/while^functional_1/convLSTM_2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2>
functional_1/convLSTM_1/whilefunctional_1/convLSTM_1/while2>
functional_1/convLSTM_2/whilefunctional_1/convLSTM_2/while:] Y
4
_output_shapes"
 :?????????x?
!
_user_specified_name	input_1
?
?
A__inference_dense_layer_call_and_return_conditional_losses_607834

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_603660

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?whilet

zeros_like	ZerosLikeinputs*
T0*<
_output_shapes*
(:&??????????????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????9M :?????????9M :?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_6032802
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_603595*
condR
while_cond_603594*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentitystrided_slice_2:output:0^StatefulPartitionedCall^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&??????????????????;O::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?	
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_603039

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????;O2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????;O:::::W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs
?f
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606107
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilew

zeros_like	ZerosLikeinputs_0*
T0*=
_output_shapes+
):'??????????????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'??????????????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_605991*
condR
while_cond_605990*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'??????????????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'??????????????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
IdentityIdentitytranspose_1:y:0^while*
T0*=
_output_shapes+
):'??????????????????v?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'??????????????????x?::2
whilewhile:g c
=
_output_shapes+
):'??????????????????x?
"
_user_specified_name
inputs/0
?X
?
while_body_604618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3~
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_4k

while/TanhTanhwhile/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2o
while/Tanh_1Tanhwhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identitywhile/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4{
while/Identity_5Identitywhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?

?
convLSTM_1_while_cond_6050692
.convlstm_1_while_convlstm_1_while_loop_counter8
4convlstm_1_while_convlstm_1_while_maximum_iterations 
convlstm_1_while_placeholder"
convlstm_1_while_placeholder_1"
convlstm_1_while_placeholder_2"
convlstm_1_while_placeholder_32
.convlstm_1_while_less_convlstm_1_strided_sliceJ
Fconvlstm_1_while_convlstm_1_while_cond_605069___redundant_placeholder0J
Fconvlstm_1_while_convlstm_1_while_cond_605069___redundant_placeholder1J
Fconvlstm_1_while_convlstm_1_while_cond_605069___redundant_placeholder2
convlstm_1_while_identity
?
convLSTM_1/while/LessLessconvlstm_1_while_placeholder.convlstm_1_while_less_convlstm_1_strided_slice*
T0*
_output_shapes
: 2
convLSTM_1/while/Less~
convLSTM_1/while/IdentityIdentityconvLSTM_1/while/Less:z:0*
T0
*
_output_shapes
: 2
convLSTM_1/while/Identity"?
convlstm_1_while_identity"convLSTM_1/while/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?	
?
-__inference_functional_1_layer_call_fn_605905

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6049252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?
?
while_cond_607061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_607061___redundant_placeholder04
0while_while_cond_607061___redundant_placeholder14
0while_while_cond_607061___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?
?
4__inference_conv_lst_m2d_cell_1_layer_call_fn_608298

inputs
states_0
states_1
unknown
	unknown_0
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????9M :?????????9M :?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_6033472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*l
_input_shapes[
Y:?????????;O:?????????9M :?????????9M ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/1
?
M
1__inference_time_distributed_layer_call_fn_606820

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:&??????????????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6028962
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'??????????????????v?:e a
=
_output_shapes+
):'??????????????????v?
 
_user_specified_nameinputs
?	
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_602984

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+???????????????????????????:::::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?H
?
__inference__traced_save_608427
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop0
,savev2_convlstm_1_kernel_read_readvariableop:
6savev2_convlstm_1_recurrent_kernel_read_readvariableop=
9savev2_time_distributed_1_moving_mean_read_readvariableopA
=savev2_time_distributed_1_moving_variance_read_readvariableop0
,savev2_convlstm_2_kernel_read_readvariableop:
6savev2_convlstm_2_recurrent_kernel_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop7
3savev2_adam_convlstm_1_kernel_m_read_readvariableopA
=savev2_adam_convlstm_1_recurrent_kernel_m_read_readvariableop7
3savev2_adam_convlstm_2_kernel_m_read_readvariableopA
=savev2_adam_convlstm_2_recurrent_kernel_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop7
3savev2_adam_convlstm_1_kernel_v_read_readvariableopA
=savev2_adam_convlstm_1_recurrent_kernel_v_read_readvariableop7
3savev2_adam_convlstm_2_kernel_v_read_readvariableopA
=savev2_adam_convlstm_2_recurrent_kernel_v_read_readvariableop
savev2_const_2

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a18d3c32e9c24295bf42dd93fee6ced1/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop,savev2_convlstm_1_kernel_read_readvariableop6savev2_convlstm_1_recurrent_kernel_read_readvariableop9savev2_time_distributed_1_moving_mean_read_readvariableop=savev2_time_distributed_1_moving_variance_read_readvariableop,savev2_convlstm_2_kernel_read_readvariableop6savev2_convlstm_2_recurrent_kernel_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop3savev2_adam_convlstm_1_kernel_m_read_readvariableop=savev2_adam_convlstm_1_recurrent_kernel_m_read_readvariableop3savev2_adam_convlstm_2_kernel_m_read_readvariableop=savev2_adam_convlstm_2_recurrent_kernel_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop3savev2_adam_convlstm_1_kernel_v_read_readvariableop=savev2_adam_convlstm_1_recurrent_kernel_v_read_readvariableop3savev2_adam_convlstm_2_kernel_v_read_readvariableop=savev2_adam_convlstm_2_recurrent_kernel_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : :: : : : : :@:@:::?: ?: : : : :?:?:?:?: ::@:@:?: ?: ::@:@:?: ?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@:,	(
&
_output_shapes
:@: 


_output_shapes
:: 

_output_shapes
::-)
'
_output_shapes
:?:-)
'
_output_shapes
: ?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:$ 

_output_shapes

: : 

_output_shapes
::,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:-)
'
_output_shapes
:?:-)
'
_output_shapes
: ?:$ 

_output_shapes

: : 

_output_shapes
::,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:- )
'
_output_shapes
:?:-!)
'
_output_shapes
: ?:"

_output_shapes
: 
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_606810

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'??????????????????v?:e a
=
_output_shapes+
):'??????????????????v?
 
_user_specified_nameinputs
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_602873

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_6028092
max_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'??????????????????v?:e a
=
_output_shapes+
):'??????????????????v?
 
_user_specified_nameinputs
?
?
A__inference_dense_layer_call_and_return_conditional_losses_604770

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?X
?
while_body_607689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3~
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_4k

while/TanhTanhwhile/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2o
while/Tanh_1Tanhwhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identitywhile/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4{
while/Identity_5Identitywhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_603023

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????;O2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????;O::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs
?
?
while_cond_606611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_606611___redundant_placeholder04
0while_while_cond_606611___redundant_placeholder14
0while_while_cond_606611___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_604617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_604617___redundant_placeholder04
0while_while_cond_604617___redundant_placeholder14
0while_while_cond_604617___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?
?
2__inference_conv_lst_m2d_cell_layer_call_fn_607993

inputs
states_0
states_1
unknown
	unknown_0
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:?????????v?:?????????v?:?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_6023122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:?????????x?:?????????v?:?????????v?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????x?
 
_user_specified_nameinputs:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/0:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/1
?e
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_604195

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :?????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_604079*
condR
while_cond_604078*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :?????????v?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????x?::2
whilewhile:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?
?
,__inference_batchnorm_1_layer_call_fn_608060

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_6029552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)functional_1_convLSTM_2_while_cond_602096L
Hfunctional_1_convlstm_2_while_functional_1_convlstm_2_while_loop_counterR
Nfunctional_1_convlstm_2_while_functional_1_convlstm_2_while_maximum_iterations-
)functional_1_convlstm_2_while_placeholder/
+functional_1_convlstm_2_while_placeholder_1/
+functional_1_convlstm_2_while_placeholder_2/
+functional_1_convlstm_2_while_placeholder_3L
Hfunctional_1_convlstm_2_while_less_functional_1_convlstm_2_strided_sliced
`functional_1_convlstm_2_while_functional_1_convlstm_2_while_cond_602096___redundant_placeholder0d
`functional_1_convlstm_2_while_functional_1_convlstm_2_while_cond_602096___redundant_placeholder1d
`functional_1_convlstm_2_while_functional_1_convlstm_2_while_cond_602096___redundant_placeholder2*
&functional_1_convlstm_2_while_identity
?
"functional_1/convLSTM_2/while/LessLess)functional_1_convlstm_2_while_placeholderHfunctional_1_convlstm_2_while_less_functional_1_convlstm_2_strided_slice*
T0*
_output_shapes
: 2$
"functional_1/convLSTM_2/while/Less?
&functional_1/convLSTM_2/while/IdentityIdentity&functional_1/convLSTM_2/while/Less:z:0*
T0
*
_output_shapes
: 2(
&functional_1/convLSTM_2/while/Identity"Y
&functional_1_convlstm_2_while_identity/functional_1/convLSTM_2/while/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?

?
convLSTM_2_while_cond_6052942
.convlstm_2_while_convlstm_2_while_loop_counter8
4convlstm_2_while_convlstm_2_while_maximum_iterations 
convlstm_2_while_placeholder"
convlstm_2_while_placeholder_1"
convlstm_2_while_placeholder_2"
convlstm_2_while_placeholder_32
.convlstm_2_while_less_convlstm_2_strided_sliceJ
Fconvlstm_2_while_convlstm_2_while_cond_605294___redundant_placeholder0J
Fconvlstm_2_while_convlstm_2_while_cond_605294___redundant_placeholder1J
Fconvlstm_2_while_convlstm_2_while_cond_605294___redundant_placeholder2
convlstm_2_while_identity
?
convLSTM_2/while/LessLessconvlstm_2_while_placeholder.convlstm_2_while_less_convlstm_2_strided_slice*
T0*
_output_shapes
: 2
convLSTM_2/while/Less~
convLSTM_2/while/IdentityIdentityconvLSTM_2/while/Less:z:0*
T0
*
_output_shapes
: 2
convLSTM_2/while/Identity"?
convlstm_2_while_identity"convLSTM_2/while/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?
J
.__inference_max_pooling2d_layer_call_fn_602815

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_6028092
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_608536
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate(
$assignvariableop_7_convlstm_1_kernel2
.assignvariableop_8_convlstm_1_recurrent_kernel5
1assignvariableop_9_time_distributed_1_moving_mean:
6assignvariableop_10_time_distributed_1_moving_variance)
%assignvariableop_11_convlstm_2_kernel3
/assignvariableop_12_convlstm_2_recurrent_kernel
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1&
"assignvariableop_17_true_positives&
"assignvariableop_18_true_negatives'
#assignvariableop_19_false_positives'
#assignvariableop_20_false_negatives+
'assignvariableop_21_adam_dense_kernel_m)
%assignvariableop_22_adam_dense_bias_m0
,assignvariableop_23_adam_convlstm_1_kernel_m:
6assignvariableop_24_adam_convlstm_1_recurrent_kernel_m0
,assignvariableop_25_adam_convlstm_2_kernel_m:
6assignvariableop_26_adam_convlstm_2_recurrent_kernel_m+
'assignvariableop_27_adam_dense_kernel_v)
%assignvariableop_28_adam_dense_bias_v0
,assignvariableop_29_adam_convlstm_1_kernel_v:
6assignvariableop_30_adam_convlstm_1_recurrent_kernel_v0
,assignvariableop_31_adam_convlstm_2_kernel_v:
6assignvariableop_32_adam_convlstm_2_recurrent_kernel_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_convlstm_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_convlstm_1_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp1assignvariableop_9_time_distributed_1_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_time_distributed_1_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_convlstm_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_convlstm_2_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_true_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_true_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp#assignvariableop_19_false_positivesIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_false_negativesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_convlstm_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_convlstm_1_recurrent_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_convlstm_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_convlstm_2_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_convlstm_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_convlstm_1_recurrent_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp,assignvariableop_31_adam_convlstm_2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_convlstm_2_recurrent_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?;
?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_602312

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constf
IdentityIdentity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:?????????x?:?????????v?:?????????v?:::X T
0
_output_shapes
:?????????x?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????v?
 
_user_specified_namestates:XT
0
_output_shapes
:?????????v?
 
_user_specified_namestates
?
,
__inference_loss_fn_0_608013
identity?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Consto
IdentityIdentity,convLSTM_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
while_cond_603594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_603594___redundant_placeholder04
0while_while_cond_603594___redundant_placeholder14
0while_while_cond_603594___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?X
?
while_body_607062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3~
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_4k

while/TanhTanhwhile/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2o
while/Tanh_1Tanhwhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identitywhile/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4{
while/Identity_5Identitywhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?

?
convLSTM_1_while_cond_6055052
.convlstm_1_while_convlstm_1_while_loop_counter8
4convlstm_1_while_convlstm_1_while_maximum_iterations 
convlstm_1_while_placeholder"
convlstm_1_while_placeholder_1"
convlstm_1_while_placeholder_2"
convlstm_1_while_placeholder_32
.convlstm_1_while_less_convlstm_1_strided_sliceJ
Fconvlstm_1_while_convlstm_1_while_cond_605505___redundant_placeholder0J
Fconvlstm_1_while_convlstm_1_while_cond_605505___redundant_placeholder1J
Fconvlstm_1_while_convlstm_1_while_cond_605505___redundant_placeholder2
convlstm_1_while_identity
?
convLSTM_1/while/LessLessconvlstm_1_while_placeholder.convlstm_1_while_less_convlstm_1_strided_slice*
T0*
_output_shapes
: 2
convLSTM_1/while/Less~
convLSTM_1/while/IdentityIdentityconvLSTM_1/while/Less:z:0*
T0
*
_output_shapes
: 2
convLSTM_1/while/Identity"?
convlstm_1_while_identity"convLSTM_1/while/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606928

inputs
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource
identity??batchnorm_1/AssignNewValue?batchnorm_1/AssignNewValue_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3Reshape:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_1/FusedBatchNormV3?
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue?
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue_1
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshape batchnorm_1/FusedBatchNormV3:y:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1?
IdentityIdentityReshape_1:output:0^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????;O::::28
batchnorm_1/AssignNewValuebatchnorm_1/AssignNewValue2<
batchnorm_1/AssignNewValue_1batchnorm_1/AssignNewValue_1:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?
?
while_cond_602622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_602622___redundant_placeholder04
0while_while_cond_602622___redundant_placeholder14
0while_while_cond_602622___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?
?
+__inference_convLSTM_2_layer_call_fn_607399
inputs_0
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6037702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&??????????????????;O::22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&??????????????????;O
"
_user_specified_name
inputs/0
?
Q
5__inference_global_max_pooling2d_layer_call_fn_603790

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_6037842
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_602809

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_604291

inputs
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource
identity?w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3Reshape:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 2
batchnorm_1/FusedBatchNormV3
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshape batchnorm_1/FusedBatchNormV3:y:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????;O:::::[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?'
?
H__inference_functional_1_layer_call_and_return_conditional_losses_604789
input_1
convlstm_1_604214
convlstm_1_604216
time_distributed_1_604318
time_distributed_1_604320
time_distributed_1_604322
time_distributed_1_604324
convlstm_2_604753
convlstm_2_604755
dense_604781
dense_604783
identity??"convLSTM_1/StatefulPartitionedCall?"convLSTM_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinput_1convlstm_1_604214convlstm_1_604216*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6039942$
"convLSTM_1/StatefulPartitionedCall?
 time_distributed/PartitionedCallPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6042272"
 time_distributed/PartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape+convLSTM_1/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0time_distributed_1_604318time_distributed_1_604320time_distributed_1_604322time_distributed_1_604324*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6042712,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape)time_distributed/PartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape?
"convLSTM_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0convlstm_2_604753convlstm_2_604755*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6045312$
"convLSTM_2/StatefulPartitionedCall?
$global_max_pooling2d/PartitionedCallPartitionedCall+convLSTM_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_6037842&
$global_max_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_604781dense_604783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6047702
dense/StatefulPartitionedCall?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^convLSTM_1/StatefulPartitionedCall#^convLSTM_2/StatefulPartitionedCall^dense/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2H
"convLSTM_2/StatefulPartitionedCall"convLSTM_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????x?
!
_user_specified_name	input_1
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_604271

inputs
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource
identity??batchnorm_1/AssignNewValue?batchnorm_1/AssignNewValue_1w
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3Reshape:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_1/FusedBatchNormV3?
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue?
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue_1
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshape batchnorm_1/FusedBatchNormV3:y:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1?
IdentityIdentityReshape_1:output:0^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????;O::::28
batchnorm_1/AssignNewValuebatchnorm_1/AssignNewValue2<
batchnorm_1/AssignNewValue_1batchnorm_1/AssignNewValue_1:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
??
?

)functional_1_convLSTM_2_while_body_602097L
Hfunctional_1_convlstm_2_while_functional_1_convlstm_2_while_loop_counterR
Nfunctional_1_convlstm_2_while_functional_1_convlstm_2_while_maximum_iterations-
)functional_1_convlstm_2_while_placeholder/
+functional_1_convlstm_2_while_placeholder_1/
+functional_1_convlstm_2_while_placeholder_2/
+functional_1_convlstm_2_while_placeholder_3I
Efunctional_1_convlstm_2_while_functional_1_convlstm_2_strided_slice_0?
?functional_1_convlstm_2_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0A
=functional_1_convlstm_2_while_split_readvariableop_resource_0C
?functional_1_convlstm_2_while_split_1_readvariableop_resource_0*
&functional_1_convlstm_2_while_identity,
(functional_1_convlstm_2_while_identity_1,
(functional_1_convlstm_2_while_identity_2,
(functional_1_convlstm_2_while_identity_3,
(functional_1_convlstm_2_while_identity_4,
(functional_1_convlstm_2_while_identity_5G
Cfunctional_1_convlstm_2_while_functional_1_convlstm_2_strided_slice?
?functional_1_convlstm_2_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_2_tensorarrayunstack_tensorlistfromtensor?
;functional_1_convlstm_2_while_split_readvariableop_resourceA
=functional_1_convlstm_2_while_split_1_readvariableop_resource??
Ofunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2Q
Ofunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Afunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?functional_1_convlstm_2_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0)functional_1_convlstm_2_while_placeholderXfunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02C
Afunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem?
#functional_1/convLSTM_2/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#functional_1/convLSTM_2/while/Const?
-functional_1/convLSTM_2/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/convLSTM_2/while/split/split_dim?
2functional_1/convLSTM_2/while/split/ReadVariableOpReadVariableOp=functional_1_convlstm_2_while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype024
2functional_1/convLSTM_2/while/split/ReadVariableOp?
#functional_1/convLSTM_2/while/splitSplit6functional_1/convLSTM_2/while/split/split_dim:output:0:functional_1/convLSTM_2/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2%
#functional_1/convLSTM_2/while/split?
%functional_1/convLSTM_2/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_2/while/Const_1?
/functional_1/convLSTM_2/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/convLSTM_2/while/split_1/split_dim?
4functional_1/convLSTM_2/while/split_1/ReadVariableOpReadVariableOp?functional_1_convlstm_2_while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype026
4functional_1/convLSTM_2/while/split_1/ReadVariableOp?
%functional_1/convLSTM_2/while/split_1Split8functional_1/convLSTM_2/while/split_1/split_dim:output:0<functional_1/convLSTM_2/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2'
%functional_1/convLSTM_2/while/split_1?
)functional_1/convLSTM_2/while/convolutionConv2DHfunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_2/while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2+
)functional_1/convLSTM_2/while/convolution?
+functional_1/convLSTM_2/while/convolution_1Conv2DHfunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_2/while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2-
+functional_1/convLSTM_2/while/convolution_1?
+functional_1/convLSTM_2/while/convolution_2Conv2DHfunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_2/while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2-
+functional_1/convLSTM_2/while/convolution_2?
+functional_1/convLSTM_2/while/convolution_3Conv2DHfunctional_1/convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_2/while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2-
+functional_1/convLSTM_2/while/convolution_3?
+functional_1/convLSTM_2/while/convolution_4Conv2D+functional_1_convlstm_2_while_placeholder_2.functional_1/convLSTM_2/while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2-
+functional_1/convLSTM_2/while/convolution_4?
+functional_1/convLSTM_2/while/convolution_5Conv2D+functional_1_convlstm_2_while_placeholder_2.functional_1/convLSTM_2/while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2-
+functional_1/convLSTM_2/while/convolution_5?
+functional_1/convLSTM_2/while/convolution_6Conv2D+functional_1_convlstm_2_while_placeholder_2.functional_1/convLSTM_2/while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2-
+functional_1/convLSTM_2/while/convolution_6?
+functional_1/convLSTM_2/while/convolution_7Conv2D+functional_1_convlstm_2_while_placeholder_2.functional_1/convLSTM_2/while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2-
+functional_1/convLSTM_2/while/convolution_7?
!functional_1/convLSTM_2/while/addAddV22functional_1/convLSTM_2/while/convolution:output:04functional_1/convLSTM_2/while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2#
!functional_1/convLSTM_2/while/add?
%functional_1/convLSTM_2/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%functional_1/convLSTM_2/while/Const_2?
%functional_1/convLSTM_2/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_2/while/Const_3?
!functional_1/convLSTM_2/while/MulMul%functional_1/convLSTM_2/while/add:z:0.functional_1/convLSTM_2/while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2#
!functional_1/convLSTM_2/while/Mul?
#functional_1/convLSTM_2/while/Add_1Add%functional_1/convLSTM_2/while/Mul:z:0.functional_1/convLSTM_2/while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/Add_1?
5functional_1/convLSTM_2/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5functional_1/convLSTM_2/while/clip_by_value/Minimum/y?
3functional_1/convLSTM_2/while/clip_by_value/MinimumMinimum'functional_1/convLSTM_2/while/Add_1:z:0>functional_1/convLSTM_2/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 25
3functional_1/convLSTM_2/while/clip_by_value/Minimum?
-functional_1/convLSTM_2/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-functional_1/convLSTM_2/while/clip_by_value/y?
+functional_1/convLSTM_2/while/clip_by_valueMaximum7functional_1/convLSTM_2/while/clip_by_value/Minimum:z:06functional_1/convLSTM_2/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2-
+functional_1/convLSTM_2/while/clip_by_value?
#functional_1/convLSTM_2/while/add_2AddV24functional_1/convLSTM_2/while/convolution_1:output:04functional_1/convLSTM_2/while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/add_2?
%functional_1/convLSTM_2/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%functional_1/convLSTM_2/while/Const_4?
%functional_1/convLSTM_2/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_2/while/Const_5?
#functional_1/convLSTM_2/while/Mul_1Mul'functional_1/convLSTM_2/while/add_2:z:0.functional_1/convLSTM_2/while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/Mul_1?
#functional_1/convLSTM_2/while/Add_3Add'functional_1/convLSTM_2/while/Mul_1:z:0.functional_1/convLSTM_2/while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/Add_3?
7functional_1/convLSTM_2/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7functional_1/convLSTM_2/while/clip_by_value_1/Minimum/y?
5functional_1/convLSTM_2/while/clip_by_value_1/MinimumMinimum'functional_1/convLSTM_2/while/Add_3:z:0@functional_1/convLSTM_2/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 27
5functional_1/convLSTM_2/while/clip_by_value_1/Minimum?
/functional_1/convLSTM_2/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/functional_1/convLSTM_2/while/clip_by_value_1/y?
-functional_1/convLSTM_2/while/clip_by_value_1Maximum9functional_1/convLSTM_2/while/clip_by_value_1/Minimum:z:08functional_1/convLSTM_2/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2/
-functional_1/convLSTM_2/while/clip_by_value_1?
#functional_1/convLSTM_2/while/mul_2Mul1functional_1/convLSTM_2/while/clip_by_value_1:z:0+functional_1_convlstm_2_while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/mul_2?
#functional_1/convLSTM_2/while/add_4AddV24functional_1/convLSTM_2/while/convolution_2:output:04functional_1/convLSTM_2/while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/add_4?
"functional_1/convLSTM_2/while/TanhTanh'functional_1/convLSTM_2/while/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2$
"functional_1/convLSTM_2/while/Tanh?
#functional_1/convLSTM_2/while/mul_3Mul/functional_1/convLSTM_2/while/clip_by_value:z:0&functional_1/convLSTM_2/while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/mul_3?
#functional_1/convLSTM_2/while/add_5AddV2'functional_1/convLSTM_2/while/mul_2:z:0'functional_1/convLSTM_2/while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/add_5?
#functional_1/convLSTM_2/while/add_6AddV24functional_1/convLSTM_2/while/convolution_3:output:04functional_1/convLSTM_2/while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/add_6?
%functional_1/convLSTM_2/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%functional_1/convLSTM_2/while/Const_6?
%functional_1/convLSTM_2/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_2/while/Const_7?
#functional_1/convLSTM_2/while/Mul_4Mul'functional_1/convLSTM_2/while/add_6:z:0.functional_1/convLSTM_2/while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/Mul_4?
#functional_1/convLSTM_2/while/Add_7Add'functional_1/convLSTM_2/while/Mul_4:z:0.functional_1/convLSTM_2/while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/Add_7?
7functional_1/convLSTM_2/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7functional_1/convLSTM_2/while/clip_by_value_2/Minimum/y?
5functional_1/convLSTM_2/while/clip_by_value_2/MinimumMinimum'functional_1/convLSTM_2/while/Add_7:z:0@functional_1/convLSTM_2/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 27
5functional_1/convLSTM_2/while/clip_by_value_2/Minimum?
/functional_1/convLSTM_2/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/functional_1/convLSTM_2/while/clip_by_value_2/y?
-functional_1/convLSTM_2/while/clip_by_value_2Maximum9functional_1/convLSTM_2/while/clip_by_value_2/Minimum:z:08functional_1/convLSTM_2/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2/
-functional_1/convLSTM_2/while/clip_by_value_2?
$functional_1/convLSTM_2/while/Tanh_1Tanh'functional_1/convLSTM_2/while/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2&
$functional_1/convLSTM_2/while/Tanh_1?
#functional_1/convLSTM_2/while/mul_5Mul1functional_1/convLSTM_2/while/clip_by_value_2:z:0(functional_1/convLSTM_2/while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2%
#functional_1/convLSTM_2/while/mul_5?
Bfunctional_1/convLSTM_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+functional_1_convlstm_2_while_placeholder_1)functional_1_convlstm_2_while_placeholder'functional_1/convLSTM_2/while/mul_5:z:0*
_output_shapes
: *
element_dtype02D
Bfunctional_1/convLSTM_2/while/TensorArrayV2Write/TensorListSetItem?
%functional_1/convLSTM_2/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_2/while/add_8/y?
#functional_1/convLSTM_2/while/add_8AddV2)functional_1_convlstm_2_while_placeholder.functional_1/convLSTM_2/while/add_8/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/convLSTM_2/while/add_8?
%functional_1/convLSTM_2/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_2/while/add_9/y?
#functional_1/convLSTM_2/while/add_9AddV2Hfunctional_1_convlstm_2_while_functional_1_convlstm_2_while_loop_counter.functional_1/convLSTM_2/while/add_9/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/convLSTM_2/while/add_9?
&functional_1/convLSTM_2/while/IdentityIdentity'functional_1/convLSTM_2/while/add_9:z:0*
T0*
_output_shapes
: 2(
&functional_1/convLSTM_2/while/Identity?
(functional_1/convLSTM_2/while/Identity_1IdentityNfunctional_1_convlstm_2_while_functional_1_convlstm_2_while_maximum_iterations*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_2/while/Identity_1?
(functional_1/convLSTM_2/while/Identity_2Identity'functional_1/convLSTM_2/while/add_8:z:0*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_2/while/Identity_2?
(functional_1/convLSTM_2/while/Identity_3IdentityRfunctional_1/convLSTM_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_2/while/Identity_3?
(functional_1/convLSTM_2/while/Identity_4Identity'functional_1/convLSTM_2/while/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2*
(functional_1/convLSTM_2/while/Identity_4?
(functional_1/convLSTM_2/while/Identity_5Identity'functional_1/convLSTM_2/while/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2*
(functional_1/convLSTM_2/while/Identity_5"?
Cfunctional_1_convlstm_2_while_functional_1_convlstm_2_strided_sliceEfunctional_1_convlstm_2_while_functional_1_convlstm_2_strided_slice_0"Y
&functional_1_convlstm_2_while_identity/functional_1/convLSTM_2/while/Identity:output:0"]
(functional_1_convlstm_2_while_identity_11functional_1/convLSTM_2/while/Identity_1:output:0"]
(functional_1_convlstm_2_while_identity_21functional_1/convLSTM_2/while/Identity_2:output:0"]
(functional_1_convlstm_2_while_identity_31functional_1/convLSTM_2/while/Identity_3:output:0"]
(functional_1_convlstm_2_while_identity_41functional_1/convLSTM_2/while/Identity_4:output:0"]
(functional_1_convlstm_2_while_identity_51functional_1/convLSTM_2/while/Identity_5:output:0"?
=functional_1_convlstm_2_while_split_1_readvariableop_resource?functional_1_convlstm_2_while_split_1_readvariableop_resource_0"|
;functional_1_convlstm_2_while_split_readvariableop_resource=functional_1_convlstm_2_while_split_readvariableop_resource_0"?
?functional_1_convlstm_2_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_2_tensorarrayunstack_tensorlistfromtensor?functional_1_convlstm_2_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?
M
1__inference_time_distributed_layer_call_fn_606769

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6042272
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????v?:\ X
4
_output_shapes"
 :?????????v?
 
_user_specified_nameinputs
?
?
,__inference_batchnorm_1_layer_call_fn_608073

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_6029842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
while_cond_604078
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_604078___redundant_placeholder04
0while_while_cond_604078___redundant_placeholder14
0while_while_cond_604078___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_604414
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_604414___redundant_placeholder04
0while_while_cond_604414___redundant_placeholder14
0while_while_cond_604414___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?f
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606308
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilew

zeros_like	ZerosLikeinputs_0*
T0*=
_output_shapes+
):'??????????????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'??????????????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_606192*
condR
while_cond_606191*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'??????????????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'??????????????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
IdentityIdentitytranspose_1:y:0^while*
T0*=
_output_shapes+
):'??????????????????v?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'??????????????????x?::2
whilewhile:g c
=
_output_shapes+
):'??????????????????x?
"
_user_specified_name
inputs/0
?'
?
H__inference_functional_1_layer_call_and_return_conditional_losses_604864

inputs
convlstm_1_604831
convlstm_1_604833
time_distributed_1_604839
time_distributed_1_604841
time_distributed_1_604843
time_distributed_1_604845
convlstm_2_604850
convlstm_2_604852
dense_604856
dense_604858
identity??"convLSTM_1/StatefulPartitionedCall?"convLSTM_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?*time_distributed_1/StatefulPartitionedCall?
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinputsconvlstm_1_604831convlstm_1_604833*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6039942$
"convLSTM_1/StatefulPartitionedCall?
 time_distributed/PartitionedCallPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_time_distributed_layer_call_and_return_conditional_losses_6042272"
 time_distributed/PartitionedCall?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshape+convLSTM_1/StatefulPartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape?
*time_distributed_1/StatefulPartitionedCallStatefulPartitionedCall)time_distributed/PartitionedCall:output:0time_distributed_1_604839time_distributed_1_604841time_distributed_1_604843time_distributed_1_604845*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6042712,
*time_distributed_1/StatefulPartitionedCall?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape)time_distributed/PartitionedCall:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape?
"convLSTM_2/StatefulPartitionedCallStatefulPartitionedCall3time_distributed_1/StatefulPartitionedCall:output:0convlstm_2_604850convlstm_2_604852*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6045312$
"convLSTM_2/StatefulPartitionedCall?
$global_max_pooling2d/PartitionedCallPartitionedCall+convLSTM_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_6037842&
$global_max_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_604856dense_604858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6047702
dense/StatefulPartitionedCall?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^convLSTM_1/StatefulPartitionedCall#^convLSTM_2/StatefulPartitionedCall^dense/StatefulPartitionedCall+^time_distributed_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2H
"convLSTM_2/StatefulPartitionedCall"convLSTM_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2X
*time_distributed_1/StatefulPartitionedCall*time_distributed_1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?
?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606851

inputs
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource
identity??batchnorm_1/AssignNewValue?batchnorm_1/AssignNewValue_1D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2	
Reshape?
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOp?
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
batchnorm_1/FusedBatchNormV3FusedBatchNormV3Reshape:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
batchnorm_1/FusedBatchNormV3?
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue?
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue_1q
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape batchnorm_1/FusedBatchNormV3:y:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1?
IdentityIdentityReshape_1:output:0^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*O
_input_shapes>
<:&??????????????????;O::::28
batchnorm_1/AssignNewValuebatchnorm_1/AssignNewValue2<
batchnorm_1/AssignNewValue_1batchnorm_1/AssignNewValue_1:d `
<
_output_shapes*
(:&??????????????????;O
 
_user_specified_nameinputs
?
?
while_cond_606191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_606191___redundant_placeholder04
0while_while_cond_606191___redundant_placeholder14
0while_while_cond_606191___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?
,
__inference_loss_fn_1_608303
identity?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Consto
IdentityIdentity,convLSTM_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?X
?
while_body_605991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:?????????v?2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_606410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_606410___redundant_placeholder04
0while_while_cond_606410___redundant_placeholder14
0while_while_cond_606410___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?X
?
while_body_607486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3~
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_4k

while/TanhTanhwhile/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2o
while/Tanh_1Tanhwhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identitywhile/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4{
while/Identity_5Identitywhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_dense_layer_call_fn_607843

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6047702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608031

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?;
?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_608268

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7{
addAddV2convolution:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Conste
IdentityIdentity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identityi

Identity_1Identity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_1i

Identity_2Identity	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*l
_input_shapes[
Y:?????????;O:?????????9M :?????????9M :::W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/1
?e
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606728

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :?????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_606612*
condR
while_cond_606611*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :?????????v?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????x?::2
whilewhile:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?

?
convLSTM_2_while_cond_6057282
.convlstm_2_while_convlstm_2_while_loop_counter8
4convlstm_2_while_convlstm_2_while_maximum_iterations 
convlstm_2_while_placeholder"
convlstm_2_while_placeholder_1"
convlstm_2_while_placeholder_2"
convlstm_2_while_placeholder_32
.convlstm_2_while_less_convlstm_2_strided_sliceJ
Fconvlstm_2_while_convlstm_2_while_cond_605728___redundant_placeholder0J
Fconvlstm_2_while_convlstm_2_while_cond_605728___redundant_placeholder1J
Fconvlstm_2_while_convlstm_2_while_cond_605728___redundant_placeholder2
convlstm_2_while_identity
?
convLSTM_2/while/LessLessconvlstm_2_while_placeholder.convlstm_2_while_less_convlstm_2_strided_slice*
T0*
_output_shapes
: 2
convLSTM_2/while/Less~
convLSTM_2/while/IdentityIdentityconvLSTM_2/while/Less:z:0*
T0
*
_output_shapes
: 2
convLSTM_2/while/Identity"?
convlstm_2_while_identity"convLSTM_2/while/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?
?
while_cond_605990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_605990___redundant_placeholder04
0while_while_cond_605990___redundant_placeholder14
0while_while_cond_605990___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*a
_input_shapesP
N: : : : :?????????v?:?????????v?: :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
:
?g
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607805

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_8}
addAddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_607689*
condR
while_cond_607688*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const|
IdentityIdentitystrided_slice_2:output:0^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????;O::2
whilewhile:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?e
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606527

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :?????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_606411*
condR
while_cond_606410*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :?????????v?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????x?::2
whilewhile:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?g
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607381
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilev

zeros_like	ZerosLikeinputs_0*
T0*<
_output_shapes*
(:&??????????????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_8}
addAddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_607265*
condR
while_cond_607264*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*<
_output_shapes*
(:&??????????????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*<
_output_shapes*
(:&??????????????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const|
IdentityIdentitystrided_slice_2:output:0^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&??????????????????;O::2
whilewhile:f b
<
_output_shapes*
(:&??????????????????;O
"
_user_specified_name
inputs/0
?X
?
while_body_604079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:?????????v?2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?n
?
convLSTM_2_while_body_6052952
.convlstm_2_while_convlstm_2_while_loop_counter8
4convlstm_2_while_convlstm_2_while_maximum_iterations 
convlstm_2_while_placeholder"
convlstm_2_while_placeholder_1"
convlstm_2_while_placeholder_2"
convlstm_2_while_placeholder_3/
+convlstm_2_while_convlstm_2_strided_slice_0m
iconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor_04
0convlstm_2_while_split_readvariableop_resource_06
2convlstm_2_while_split_1_readvariableop_resource_0
convlstm_2_while_identity
convlstm_2_while_identity_1
convlstm_2_while_identity_2
convlstm_2_while_identity_3
convlstm_2_while_identity_4
convlstm_2_while_identity_5-
)convlstm_2_while_convlstm_2_strided_slicek
gconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor2
.convlstm_2_while_split_readvariableop_resource4
0convlstm_2_while_split_1_readvariableop_resource??
BconvLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2D
BconvLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
4convLSTM_2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0convlstm_2_while_placeholderKconvLSTM_2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype026
4convLSTM_2/while/TensorArrayV2Read/TensorListGetItemr
convLSTM_2/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/Const?
 convLSTM_2/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_2/while/split/split_dim?
%convLSTM_2/while/split/ReadVariableOpReadVariableOp0convlstm_2_while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02'
%convLSTM_2/while/split/ReadVariableOp?
convLSTM_2/while/splitSplit)convLSTM_2/while/split/split_dim:output:0-convLSTM_2/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
convLSTM_2/while/splitv
convLSTM_2/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/Const_1?
"convLSTM_2/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"convLSTM_2/while/split_1/split_dim?
'convLSTM_2/while/split_1/ReadVariableOpReadVariableOp2convlstm_2_while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02)
'convLSTM_2/while/split_1/ReadVariableOp?
convLSTM_2/while/split_1Split+convLSTM_2/while/split_1/split_dim:output:0/convLSTM_2/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
convLSTM_2/while/split_1?
convLSTM_2/while/convolutionConv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/while/convolution?
convLSTM_2/while/convolution_1Conv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2 
convLSTM_2/while/convolution_1?
convLSTM_2/while/convolution_2Conv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2 
convLSTM_2/while/convolution_2?
convLSTM_2/while/convolution_3Conv2D;convLSTM_2/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_2/while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2 
convLSTM_2/while/convolution_3?
convLSTM_2/while/convolution_4Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_4?
convLSTM_2/while/convolution_5Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_5?
convLSTM_2/while/convolution_6Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_6?
convLSTM_2/while/convolution_7Conv2Dconvlstm_2_while_placeholder_2!convLSTM_2/while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2 
convLSTM_2/while/convolution_7?
convLSTM_2/while/addAddV2%convLSTM_2/while/convolution:output:0'convLSTM_2/while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/addy
convLSTM_2/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/while/Const_2y
convLSTM_2/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/while/Const_3?
convLSTM_2/while/MulMulconvLSTM_2/while/add:z:0!convLSTM_2/while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Mul?
convLSTM_2/while/Add_1AddconvLSTM_2/while/Mul:z:0!convLSTM_2/while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Add_1?
(convLSTM_2/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(convLSTM_2/while/clip_by_value/Minimum/y?
&convLSTM_2/while/clip_by_value/MinimumMinimumconvLSTM_2/while/Add_1:z:01convLSTM_2/while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2(
&convLSTM_2/while/clip_by_value/Minimum?
 convLSTM_2/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 convLSTM_2/while/clip_by_value/y?
convLSTM_2/while/clip_by_valueMaximum*convLSTM_2/while/clip_by_value/Minimum:z:0)convLSTM_2/while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2 
convLSTM_2/while/clip_by_value?
convLSTM_2/while/add_2AddV2'convLSTM_2/while/convolution_1:output:0'convLSTM_2/while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_2y
convLSTM_2/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/while/Const_4y
convLSTM_2/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/while/Const_5?
convLSTM_2/while/Mul_1MulconvLSTM_2/while/add_2:z:0!convLSTM_2/while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Mul_1?
convLSTM_2/while/Add_3AddconvLSTM_2/while/Mul_1:z:0!convLSTM_2/while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Add_3?
*convLSTM_2/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_2/while/clip_by_value_1/Minimum/y?
(convLSTM_2/while/clip_by_value_1/MinimumMinimumconvLSTM_2/while/Add_3:z:03convLSTM_2/while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2*
(convLSTM_2/while/clip_by_value_1/Minimum?
"convLSTM_2/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_2/while/clip_by_value_1/y?
 convLSTM_2/while/clip_by_value_1Maximum,convLSTM_2/while/clip_by_value_1/Minimum:z:0+convLSTM_2/while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2"
 convLSTM_2/while/clip_by_value_1?
convLSTM_2/while/mul_2Mul$convLSTM_2/while/clip_by_value_1:z:0convlstm_2_while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/mul_2?
convLSTM_2/while/add_4AddV2'convLSTM_2/while/convolution_2:output:0'convLSTM_2/while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_4?
convLSTM_2/while/TanhTanhconvLSTM_2/while/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Tanh?
convLSTM_2/while/mul_3Mul"convLSTM_2/while/clip_by_value:z:0convLSTM_2/while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/mul_3?
convLSTM_2/while/add_5AddV2convLSTM_2/while/mul_2:z:0convLSTM_2/while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_5?
convLSTM_2/while/add_6AddV2'convLSTM_2/while/convolution_3:output:0'convLSTM_2/while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/add_6y
convLSTM_2/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/while/Const_6y
convLSTM_2/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/while/Const_7?
convLSTM_2/while/Mul_4MulconvLSTM_2/while/add_6:z:0!convLSTM_2/while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Mul_4?
convLSTM_2/while/Add_7AddconvLSTM_2/while/Mul_4:z:0!convLSTM_2/while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Add_7?
*convLSTM_2/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2,
*convLSTM_2/while/clip_by_value_2/Minimum/y?
(convLSTM_2/while/clip_by_value_2/MinimumMinimumconvLSTM_2/while/Add_7:z:03convLSTM_2/while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2*
(convLSTM_2/while/clip_by_value_2/Minimum?
"convLSTM_2/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_2/while/clip_by_value_2/y?
 convLSTM_2/while/clip_by_value_2Maximum,convLSTM_2/while/clip_by_value_2/Minimum:z:0+convLSTM_2/while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2"
 convLSTM_2/while/clip_by_value_2?
convLSTM_2/while/Tanh_1TanhconvLSTM_2/while/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Tanh_1?
convLSTM_2/while/mul_5Mul$convLSTM_2/while/clip_by_value_2:z:0convLSTM_2/while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/mul_5?
5convLSTM_2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemconvlstm_2_while_placeholder_1convlstm_2_while_placeholderconvLSTM_2/while/mul_5:z:0*
_output_shapes
: *
element_dtype027
5convLSTM_2/while/TensorArrayV2Write/TensorListSetItemv
convLSTM_2/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/add_8/y?
convLSTM_2/while/add_8AddV2convlstm_2_while_placeholder!convLSTM_2/while/add_8/y:output:0*
T0*
_output_shapes
: 2
convLSTM_2/while/add_8v
convLSTM_2/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/while/add_9/y?
convLSTM_2/while/add_9AddV2.convlstm_2_while_convlstm_2_while_loop_counter!convLSTM_2/while/add_9/y:output:0*
T0*
_output_shapes
: 2
convLSTM_2/while/add_9
convLSTM_2/while/IdentityIdentityconvLSTM_2/while/add_9:z:0*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity?
convLSTM_2/while/Identity_1Identity4convlstm_2_while_convlstm_2_while_maximum_iterations*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity_1?
convLSTM_2/while/Identity_2IdentityconvLSTM_2/while/add_8:z:0*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity_2?
convLSTM_2/while/Identity_3IdentityEconvLSTM_2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
convLSTM_2/while/Identity_3?
convLSTM_2/while/Identity_4IdentityconvLSTM_2/while/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Identity_4?
convLSTM_2/while/Identity_5IdentityconvLSTM_2/while/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/while/Identity_5"X
)convlstm_2_while_convlstm_2_strided_slice+convlstm_2_while_convlstm_2_strided_slice_0"?
convlstm_2_while_identity"convLSTM_2/while/Identity:output:0"C
convlstm_2_while_identity_1$convLSTM_2/while/Identity_1:output:0"C
convlstm_2_while_identity_2$convLSTM_2/while/Identity_2:output:0"C
convlstm_2_while_identity_3$convLSTM_2/while/Identity_3:output:0"C
convlstm_2_while_identity_4$convLSTM_2/while/Identity_4:output:0"C
convlstm_2_while_identity_5$convLSTM_2/while/Identity_5:output:0"f
0convlstm_2_while_split_1_readvariableop_resource2convlstm_2_while_split_1_readvariableop_resource_0"b
.convlstm_2_while_split_readvariableop_resource0convlstm_2_while_split_readvariableop_resource_0"?
gconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensoriconvlstm_2_while_tensorarrayv2read_tensorlistgetitem_convlstm_2_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_604985
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_6022212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????x?
!
_user_specified_name	input_1
?
?
while_cond_603704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_603704___redundant_placeholder04
0while_while_cond_603704___redundant_placeholder14
0while_while_cond_603704___redundant_placeholder2
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*_
_input_shapesN
L: : : : :?????????9M :?????????9M : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
:
?;
?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_608201

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7{
addAddV2convolution:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Conste
IdentityIdentity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identityi

Identity_1Identity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_1i

Identity_2Identity	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*l
_input_shapes[
Y:?????????;O:?????????9M :?????????9M :::W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/0:YU
/
_output_shapes
:?????????9M 
"
_user_specified_name
states/1
?
?
3__inference_time_distributed_1_layer_call_fn_606961

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_6042712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????;O::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?	
?
-__inference_functional_1_layer_call_fn_604887
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6048642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????x?
!
_user_specified_name	input_1
?
l
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_603784

inputs
identity
Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Max/reduction_indicest
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_604227

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????v?:\ X
4
_output_shapes"
 :?????????v?
 
_user_specified_nameinputs
?e
?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_603994

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :?????????x?2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices|
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_603878*
condR
while_cond_603877*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2
transpose_1?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constx
IdentityIdentitytranspose_1:y:0^while*
T0*4
_output_shapes"
 :?????????v?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????x?::2
whilewhile:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?	
?
-__inference_functional_1_layer_call_fn_604948
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_6049252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :?????????x?
!
_user_specified_name	input_1
?
?
+__inference_convLSTM_1_layer_call_fn_606326
inputs_0
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'??????????????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6027962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'??????????????????v?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'??????????????????x?::22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'??????????????????x?
"
_user_specified_name
inputs/0
??
?
H__inference_functional_1_layer_call_and_return_conditional_losses_605855

inputs,
(convlstm_1_split_readvariableop_resource.
*convlstm_1_split_1_readvariableop_resource(
$time_distributed_1_batchnorm_1_scale)
%time_distributed_1_batchnorm_1_offsetK
Gtime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resourceM
Itime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource,
(convlstm_2_split_readvariableop_resource.
*convlstm_2_split_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??convLSTM_1/while?convLSTM_2/while?
convLSTM_1/zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :?????????x?2
convLSTM_1/zeros_like?
 convLSTM_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/Sum/reduction_indices?
convLSTM_1/SumSumconvLSTM_1/zeros_like:y:0)convLSTM_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
convLSTM_1/Sum?
convLSTM_1/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
convLSTM_1/zeros?
convLSTM_1/convolutionConv2DconvLSTM_1/Sum:output:0convLSTM_1/zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution?
convLSTM_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose/perm?
convLSTM_1/transpose	Transposeinputs"convLSTM_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2
convLSTM_1/transposel
convLSTM_1/ShapeShapeconvLSTM_1/transpose:y:0*
T0*
_output_shapes
:2
convLSTM_1/Shape?
convLSTM_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
convLSTM_1/strided_slice/stack?
 convLSTM_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_1?
 convLSTM_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_2?
convLSTM_1/strided_sliceStridedSliceconvLSTM_1/Shape:output:0'convLSTM_1/strided_slice/stack:output:0)convLSTM_1/strided_slice/stack_1:output:0)convLSTM_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
convLSTM_1/strided_slice?
&convLSTM_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&convLSTM_1/TensorArrayV2/element_shape?
convLSTM_1/TensorArrayV2TensorListReserve/convLSTM_1/TensorArrayV2/element_shape:output:0!convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_1/TensorArrayV2?
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      2B
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
2convLSTM_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconvLSTM_1/transpose:y:0IconvLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2convLSTM_1/TensorArrayUnstack/TensorListFromTensor?
 convLSTM_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 convLSTM_1/strided_slice_1/stack?
"convLSTM_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_1?
"convLSTM_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_2?
convLSTM_1/strided_slice_1StridedSliceconvLSTM_1/transpose:y:0)convLSTM_1/strided_slice_1/stack:output:0+convLSTM_1/strided_slice_1/stack_1:output:0+convLSTM_1/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
convLSTM_1/strided_slice_1f
convLSTM_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/Constz
convLSTM_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/split/split_dim?
convLSTM_1/split/ReadVariableOpReadVariableOp(convlstm_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype02!
convLSTM_1/split/ReadVariableOp?
convLSTM_1/splitSplit#convLSTM_1/split/split_dim:output:0'convLSTM_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/splitj
convLSTM_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/Const_1~
convLSTM_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/split_1/split_dim?
!convLSTM_1/split_1/ReadVariableOpReadVariableOp*convlstm_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!convLSTM_1/split_1/ReadVariableOp?
convLSTM_1/split_1Split%convLSTM_1/split_1/split_dim:output:0)convLSTM_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/split_1?
convLSTM_1/convolution_1Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_1?
convLSTM_1/convolution_2Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_2?
convLSTM_1/convolution_3Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_3?
convLSTM_1/convolution_4Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_4?
convLSTM_1/convolution_5Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_5?
convLSTM_1/convolution_6Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_6?
convLSTM_1/convolution_7Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_7?
convLSTM_1/convolution_8Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_8?
convLSTM_1/addAddV2!convLSTM_1/convolution_1:output:0!convLSTM_1/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/addm
convLSTM_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/Const_2m
convLSTM_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_3?
convLSTM_1/MulMulconvLSTM_1/add:z:0convLSTM_1/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Mul?
convLSTM_1/Add_1AddconvLSTM_1/Mul:z:0convLSTM_1/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Add_1?
"convLSTM_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"convLSTM_1/clip_by_value/Minimum/y?
 convLSTM_1/clip_by_value/MinimumMinimumconvLSTM_1/Add_1:z:0+convLSTM_1/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2"
 convLSTM_1/clip_by_value/Minimum}
convLSTM_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value/y?
convLSTM_1/clip_by_valueMaximum$convLSTM_1/clip_by_value/Minimum:z:0#convLSTM_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/clip_by_value?
convLSTM_1/add_2AddV2!convLSTM_1/convolution_2:output:0!convLSTM_1/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_2m
convLSTM_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/Const_4m
convLSTM_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_5?
convLSTM_1/Mul_1MulconvLSTM_1/add_2:z:0convLSTM_1/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Mul_1?
convLSTM_1/Add_3AddconvLSTM_1/Mul_1:z:0convLSTM_1/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Add_3?
$convLSTM_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_1/clip_by_value_1/Minimum/y?
"convLSTM_1/clip_by_value_1/MinimumMinimumconvLSTM_1/Add_3:z:0-convLSTM_1/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2$
"convLSTM_1/clip_by_value_1/Minimum?
convLSTM_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_1/y?
convLSTM_1/clip_by_value_1Maximum&convLSTM_1/clip_by_value_1/Minimum:z:0%convLSTM_1/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/clip_by_value_1?
convLSTM_1/mul_2MulconvLSTM_1/clip_by_value_1:z:0convLSTM_1/convolution:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/mul_2?
convLSTM_1/add_4AddV2!convLSTM_1/convolution_3:output:0!convLSTM_1/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_4{
convLSTM_1/TanhTanhconvLSTM_1/add_4:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Tanh?
convLSTM_1/mul_3MulconvLSTM_1/clip_by_value:z:0convLSTM_1/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/mul_3?
convLSTM_1/add_5AddV2convLSTM_1/mul_2:z:0convLSTM_1/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_5?
convLSTM_1/add_6AddV2!convLSTM_1/convolution_4:output:0!convLSTM_1/convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_6m
convLSTM_1/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/Const_6m
convLSTM_1/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_7?
convLSTM_1/Mul_4MulconvLSTM_1/add_6:z:0convLSTM_1/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Mul_4?
convLSTM_1/Add_7AddconvLSTM_1/Mul_4:z:0convLSTM_1/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Add_7?
$convLSTM_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_1/clip_by_value_2/Minimum/y?
"convLSTM_1/clip_by_value_2/MinimumMinimumconvLSTM_1/Add_7:z:0-convLSTM_1/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2$
"convLSTM_1/clip_by_value_2/Minimum?
convLSTM_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_2/y?
convLSTM_1/clip_by_value_2Maximum&convLSTM_1/clip_by_value_2/Minimum:z:0%convLSTM_1/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/clip_by_value_2
convLSTM_1/Tanh_1TanhconvLSTM_1/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Tanh_1?
convLSTM_1/mul_5MulconvLSTM_1/clip_by_value_2:z:0convLSTM_1/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/mul_5?
(convLSTM_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2*
(convLSTM_1/TensorArrayV2_1/element_shape?
convLSTM_1/TensorArrayV2_1TensorListReserve1convLSTM_1/TensorArrayV2_1/element_shape:output:0!convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_1/TensorArrayV2_1d
convLSTM_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_1/time?
#convLSTM_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#convLSTM_1/while/maximum_iterations?
convLSTM_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_1/while/loop_counter?
convLSTM_1/whileWhile&convLSTM_1/while/loop_counter:output:0,convLSTM_1/while/maximum_iterations:output:0convLSTM_1/time:output:0#convLSTM_1/TensorArrayV2_1:handle:0convLSTM_1/convolution:output:0convLSTM_1/convolution:output:0!convLSTM_1/strided_slice:output:0BconvLSTM_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(convlstm_1_split_readvariableop_resource*convlstm_1_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*(
body R
convLSTM_1_while_body_605506*(
cond R
convLSTM_1_while_cond_605505*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
convLSTM_1/while?
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2=
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape?
-convLSTM_1/TensorArrayV2Stack/TensorListStackTensorListStackconvLSTM_1/while:output:3DconvLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02/
-convLSTM_1/TensorArrayV2Stack/TensorListStack?
 convLSTM_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 convLSTM_1/strided_slice_2/stack?
"convLSTM_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"convLSTM_1/strided_slice_2/stack_1?
"convLSTM_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_2/stack_2?
convLSTM_1/strided_slice_2StridedSlice6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0)convLSTM_1/strided_slice_2/stack:output:0+convLSTM_1/strided_slice_2/stack_1:output:0+convLSTM_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
convLSTM_1/strided_slice_2?
convLSTM_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose_1/perm?
convLSTM_1/transpose_1	Transpose6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0$convLSTM_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2
convLSTM_1/transpose_1?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapeconvLSTM_1/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape?
&time_distributed/max_pooling2d/MaxPoolMaxPool!time_distributed/Reshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2(
&time_distributed/max_pooling2d/MaxPool?
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2"
 time_distributed/Reshape_1/shape?
time_distributed/Reshape_1Reshape/time_distributed/max_pooling2d/MaxPool:output:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
time_distributed/Reshape_1?
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2"
 time_distributed/Reshape_2/shape?
time_distributed/Reshape_2ReshapeconvLSTM_1/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape_2?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape?
>time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOpGtime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp?
@time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpItime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
/time_distributed_1/batchnorm_1/FusedBatchNormV3FusedBatchNormV3#time_distributed_1/Reshape:output:0$time_distributed_1_batchnorm_1_scale%time_distributed_1_batchnorm_1_offsetFtime_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0Htime_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
is_training( 21
/time_distributed_1/batchnorm_1/FusedBatchNormV3?
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2$
"time_distributed_1/Reshape_1/shape?
time_distributed_1/Reshape_1Reshape3time_distributed_1/batchnorm_1/FusedBatchNormV3:y:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
time_distributed_1/Reshape_1?
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2$
"time_distributed_1/Reshape_2/shape?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape_2?
convLSTM_2/zeros_like	ZerosLike%time_distributed_1/Reshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2
convLSTM_2/zeros_like?
 convLSTM_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_2/Sum/reduction_indices?
convLSTM_2/SumSumconvLSTM_2/zeros_like:y:0)convLSTM_2/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
convLSTM_2/Sum?
 convLSTM_2/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 convLSTM_2/zeros/shape_as_tensoru
convLSTM_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/zeros/Const?
convLSTM_2/zerosFill)convLSTM_2/zeros/shape_as_tensor:output:0convLSTM_2/zeros/Const:output:0*
T0*&
_output_shapes
: 2
convLSTM_2/zeros?
convLSTM_2/convolutionConv2DconvLSTM_2/Sum:output:0convLSTM_2/zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution?
convLSTM_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_2/transpose/perm?
convLSTM_2/transpose	Transpose%time_distributed_1/Reshape_1:output:0"convLSTM_2/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2
convLSTM_2/transposel
convLSTM_2/ShapeShapeconvLSTM_2/transpose:y:0*
T0*
_output_shapes
:2
convLSTM_2/Shape?
convLSTM_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
convLSTM_2/strided_slice/stack?
 convLSTM_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_2/strided_slice/stack_1?
 convLSTM_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_2/strided_slice/stack_2?
convLSTM_2/strided_sliceStridedSliceconvLSTM_2/Shape:output:0'convLSTM_2/strided_slice/stack:output:0)convLSTM_2/strided_slice/stack_1:output:0)convLSTM_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
convLSTM_2/strided_slice?
&convLSTM_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&convLSTM_2/TensorArrayV2/element_shape?
convLSTM_2/TensorArrayV2TensorListReserve/convLSTM_2/TensorArrayV2/element_shape:output:0!convLSTM_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_2/TensorArrayV2?
@convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2B
@convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
2convLSTM_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconvLSTM_2/transpose:y:0IconvLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2convLSTM_2/TensorArrayUnstack/TensorListFromTensor?
 convLSTM_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 convLSTM_2/strided_slice_1/stack?
"convLSTM_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_2/strided_slice_1/stack_1?
"convLSTM_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_2/strided_slice_1/stack_2?
convLSTM_2/strided_slice_1StridedSliceconvLSTM_2/transpose:y:0)convLSTM_2/strided_slice_1/stack:output:0+convLSTM_2/strided_slice_1/stack_1:output:0+convLSTM_2/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
convLSTM_2/strided_slice_1f
convLSTM_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/Constz
convLSTM_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/split/split_dim?
convLSTM_2/split/ReadVariableOpReadVariableOp(convlstm_2_split_readvariableop_resource*'
_output_shapes
:?*
dtype02!
convLSTM_2/split/ReadVariableOp?
convLSTM_2/splitSplit#convLSTM_2/split/split_dim:output:0'convLSTM_2/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
convLSTM_2/splitj
convLSTM_2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/Const_1~
convLSTM_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/split_1/split_dim?
!convLSTM_2/split_1/ReadVariableOpReadVariableOp*convlstm_2_split_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02#
!convLSTM_2/split_1/ReadVariableOp?
convLSTM_2/split_1Split%convLSTM_2/split_1/split_dim:output:0)convLSTM_2/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
convLSTM_2/split_1?
convLSTM_2/convolution_1Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_1?
convLSTM_2/convolution_2Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_2?
convLSTM_2/convolution_3Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_3?
convLSTM_2/convolution_4Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_4?
convLSTM_2/convolution_5Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_5?
convLSTM_2/convolution_6Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_6?
convLSTM_2/convolution_7Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_7?
convLSTM_2/convolution_8Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_8?
convLSTM_2/addAddV2!convLSTM_2/convolution_1:output:0!convLSTM_2/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/addm
convLSTM_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/Const_2m
convLSTM_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/Const_3?
convLSTM_2/MulMulconvLSTM_2/add:z:0convLSTM_2/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Mul?
convLSTM_2/Add_1AddconvLSTM_2/Mul:z:0convLSTM_2/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Add_1?
"convLSTM_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"convLSTM_2/clip_by_value/Minimum/y?
 convLSTM_2/clip_by_value/MinimumMinimumconvLSTM_2/Add_1:z:0+convLSTM_2/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2"
 convLSTM_2/clip_by_value/Minimum}
convLSTM_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/clip_by_value/y?
convLSTM_2/clip_by_valueMaximum$convLSTM_2/clip_by_value/Minimum:z:0#convLSTM_2/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/clip_by_value?
convLSTM_2/add_2AddV2!convLSTM_2/convolution_2:output:0!convLSTM_2/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_2m
convLSTM_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/Const_4m
convLSTM_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/Const_5?
convLSTM_2/Mul_1MulconvLSTM_2/add_2:z:0convLSTM_2/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Mul_1?
convLSTM_2/Add_3AddconvLSTM_2/Mul_1:z:0convLSTM_2/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Add_3?
$convLSTM_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_2/clip_by_value_1/Minimum/y?
"convLSTM_2/clip_by_value_1/MinimumMinimumconvLSTM_2/Add_3:z:0-convLSTM_2/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2$
"convLSTM_2/clip_by_value_1/Minimum?
convLSTM_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/clip_by_value_1/y?
convLSTM_2/clip_by_value_1Maximum&convLSTM_2/clip_by_value_1/Minimum:z:0%convLSTM_2/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/clip_by_value_1?
convLSTM_2/mul_2MulconvLSTM_2/clip_by_value_1:z:0convLSTM_2/convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/mul_2?
convLSTM_2/add_4AddV2!convLSTM_2/convolution_3:output:0!convLSTM_2/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_4z
convLSTM_2/TanhTanhconvLSTM_2/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Tanh?
convLSTM_2/mul_3MulconvLSTM_2/clip_by_value:z:0convLSTM_2/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/mul_3?
convLSTM_2/add_5AddV2convLSTM_2/mul_2:z:0convLSTM_2/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_5?
convLSTM_2/add_6AddV2!convLSTM_2/convolution_4:output:0!convLSTM_2/convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_6m
convLSTM_2/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/Const_6m
convLSTM_2/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/Const_7?
convLSTM_2/Mul_4MulconvLSTM_2/add_6:z:0convLSTM_2/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Mul_4?
convLSTM_2/Add_7AddconvLSTM_2/Mul_4:z:0convLSTM_2/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Add_7?
$convLSTM_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_2/clip_by_value_2/Minimum/y?
"convLSTM_2/clip_by_value_2/MinimumMinimumconvLSTM_2/Add_7:z:0-convLSTM_2/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2$
"convLSTM_2/clip_by_value_2/Minimum?
convLSTM_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/clip_by_value_2/y?
convLSTM_2/clip_by_value_2Maximum&convLSTM_2/clip_by_value_2/Minimum:z:0%convLSTM_2/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/clip_by_value_2~
convLSTM_2/Tanh_1TanhconvLSTM_2/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Tanh_1?
convLSTM_2/mul_5MulconvLSTM_2/clip_by_value_2:z:0convLSTM_2/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/mul_5?
(convLSTM_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2*
(convLSTM_2/TensorArrayV2_1/element_shape?
convLSTM_2/TensorArrayV2_1TensorListReserve1convLSTM_2/TensorArrayV2_1/element_shape:output:0!convLSTM_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_2/TensorArrayV2_1d
convLSTM_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_2/time?
#convLSTM_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#convLSTM_2/while/maximum_iterations?
convLSTM_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_2/while/loop_counter?
convLSTM_2/whileWhile&convLSTM_2/while/loop_counter:output:0,convLSTM_2/while/maximum_iterations:output:0convLSTM_2/time:output:0#convLSTM_2/TensorArrayV2_1:handle:0convLSTM_2/convolution:output:0convLSTM_2/convolution:output:0!convLSTM_2/strided_slice:output:0BconvLSTM_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0(convlstm_2_split_readvariableop_resource*convlstm_2_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*(
body R
convLSTM_2_while_body_605729*(
cond R
convLSTM_2_while_cond_605728*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
convLSTM_2/while?
;convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2=
;convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shape?
-convLSTM_2/TensorArrayV2Stack/TensorListStackTensorListStackconvLSTM_2/while:output:3DconvLSTM_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02/
-convLSTM_2/TensorArrayV2Stack/TensorListStack?
 convLSTM_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 convLSTM_2/strided_slice_2/stack?
"convLSTM_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"convLSTM_2/strided_slice_2/stack_1?
"convLSTM_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_2/strided_slice_2/stack_2?
convLSTM_2/strided_slice_2StridedSlice6convLSTM_2/TensorArrayV2Stack/TensorListStack:tensor:0)convLSTM_2/strided_slice_2/stack:output:0+convLSTM_2/strided_slice_2/stack_1:output:0+convLSTM_2/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
convLSTM_2/strided_slice_2?
convLSTM_2/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_2/transpose_1/perm?
convLSTM_2/transpose_1	Transpose6convLSTM_2/TensorArrayV2Stack/TensorListStack:tensor:0$convLSTM_2/transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2
convLSTM_2/transpose_1?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax#convLSTM_2/strided_slice_2:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
global_max_pooling2d/Max?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul!global_max_pooling2d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentitydense/Sigmoid:y:0^convLSTM_1/while^convLSTM_2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2$
convLSTM_1/whileconvLSTM_1/while2$
convLSTM_2/whileconvLSTM_2/while:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
լ
?
H__inference_functional_1_layer_call_and_return_conditional_losses_605421

inputs,
(convlstm_1_split_readvariableop_resource.
*convlstm_1_split_1_readvariableop_resource(
$time_distributed_1_batchnorm_1_scale)
%time_distributed_1_batchnorm_1_offsetK
Gtime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resourceM
Itime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource,
(convlstm_2_split_readvariableop_resource.
*convlstm_2_split_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??convLSTM_1/while?convLSTM_2/while?-time_distributed_1/batchnorm_1/AssignNewValue?/time_distributed_1/batchnorm_1/AssignNewValue_1?
convLSTM_1/zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :?????????x?2
convLSTM_1/zeros_like?
 convLSTM_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/Sum/reduction_indices?
convLSTM_1/SumSumconvLSTM_1/zeros_like:y:0)convLSTM_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:?????????x?2
convLSTM_1/Sum?
convLSTM_1/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
convLSTM_1/zeros?
convLSTM_1/convolutionConv2DconvLSTM_1/Sum:output:0convLSTM_1/zeros:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution?
convLSTM_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose/perm?
convLSTM_1/transpose	Transposeinputs"convLSTM_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :?????????x?2
convLSTM_1/transposel
convLSTM_1/ShapeShapeconvLSTM_1/transpose:y:0*
T0*
_output_shapes
:2
convLSTM_1/Shape?
convLSTM_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
convLSTM_1/strided_slice/stack?
 convLSTM_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_1?
 convLSTM_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_2?
convLSTM_1/strided_sliceStridedSliceconvLSTM_1/Shape:output:0'convLSTM_1/strided_slice/stack:output:0)convLSTM_1/strided_slice/stack_1:output:0)convLSTM_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
convLSTM_1/strided_slice?
&convLSTM_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&convLSTM_1/TensorArrayV2/element_shape?
convLSTM_1/TensorArrayV2TensorListReserve/convLSTM_1/TensorArrayV2/element_shape:output:0!convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_1/TensorArrayV2?
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      2B
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape?
2convLSTM_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconvLSTM_1/transpose:y:0IconvLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2convLSTM_1/TensorArrayUnstack/TensorListFromTensor?
 convLSTM_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 convLSTM_1/strided_slice_1/stack?
"convLSTM_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_1?
"convLSTM_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_2?
convLSTM_1/strided_slice_1StridedSliceconvLSTM_1/transpose:y:0)convLSTM_1/strided_slice_1/stack:output:0+convLSTM_1/strided_slice_1/stack_1:output:0+convLSTM_1/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????x?*
shrink_axis_mask2
convLSTM_1/strided_slice_1f
convLSTM_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/Constz
convLSTM_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/split/split_dim?
convLSTM_1/split/ReadVariableOpReadVariableOp(convlstm_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype02!
convLSTM_1/split/ReadVariableOp?
convLSTM_1/splitSplit#convLSTM_1/split/split_dim:output:0'convLSTM_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/splitj
convLSTM_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/Const_1~
convLSTM_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/split_1/split_dim?
!convLSTM_1/split_1/ReadVariableOpReadVariableOp*convlstm_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!convLSTM_1/split_1/ReadVariableOp?
convLSTM_1/split_1Split%convLSTM_1/split_1/split_dim:output:0)convLSTM_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/split_1?
convLSTM_1/convolution_1Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_1?
convLSTM_1/convolution_2Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_2?
convLSTM_1/convolution_3Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_3?
convLSTM_1/convolution_4Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convLSTM_1/convolution_4?
convLSTM_1/convolution_5Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_5?
convLSTM_1/convolution_6Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_6?
convLSTM_1/convolution_7Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_7?
convLSTM_1/convolution_8Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convLSTM_1/convolution_8?
convLSTM_1/addAddV2!convLSTM_1/convolution_1:output:0!convLSTM_1/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/addm
convLSTM_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/Const_2m
convLSTM_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_3?
convLSTM_1/MulMulconvLSTM_1/add:z:0convLSTM_1/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Mul?
convLSTM_1/Add_1AddconvLSTM_1/Mul:z:0convLSTM_1/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Add_1?
"convLSTM_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"convLSTM_1/clip_by_value/Minimum/y?
 convLSTM_1/clip_by_value/MinimumMinimumconvLSTM_1/Add_1:z:0+convLSTM_1/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2"
 convLSTM_1/clip_by_value/Minimum}
convLSTM_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value/y?
convLSTM_1/clip_by_valueMaximum$convLSTM_1/clip_by_value/Minimum:z:0#convLSTM_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/clip_by_value?
convLSTM_1/add_2AddV2!convLSTM_1/convolution_2:output:0!convLSTM_1/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_2m
convLSTM_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/Const_4m
convLSTM_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_5?
convLSTM_1/Mul_1MulconvLSTM_1/add_2:z:0convLSTM_1/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Mul_1?
convLSTM_1/Add_3AddconvLSTM_1/Mul_1:z:0convLSTM_1/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Add_3?
$convLSTM_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_1/clip_by_value_1/Minimum/y?
"convLSTM_1/clip_by_value_1/MinimumMinimumconvLSTM_1/Add_3:z:0-convLSTM_1/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2$
"convLSTM_1/clip_by_value_1/Minimum?
convLSTM_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_1/y?
convLSTM_1/clip_by_value_1Maximum&convLSTM_1/clip_by_value_1/Minimum:z:0%convLSTM_1/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/clip_by_value_1?
convLSTM_1/mul_2MulconvLSTM_1/clip_by_value_1:z:0convLSTM_1/convolution:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/mul_2?
convLSTM_1/add_4AddV2!convLSTM_1/convolution_3:output:0!convLSTM_1/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_4{
convLSTM_1/TanhTanhconvLSTM_1/add_4:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Tanh?
convLSTM_1/mul_3MulconvLSTM_1/clip_by_value:z:0convLSTM_1/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/mul_3?
convLSTM_1/add_5AddV2convLSTM_1/mul_2:z:0convLSTM_1/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_5?
convLSTM_1/add_6AddV2!convLSTM_1/convolution_4:output:0!convLSTM_1/convolution_8:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/add_6m
convLSTM_1/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_1/Const_6m
convLSTM_1/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_7?
convLSTM_1/Mul_4MulconvLSTM_1/add_6:z:0convLSTM_1/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Mul_4?
convLSTM_1/Add_7AddconvLSTM_1/Mul_4:z:0convLSTM_1/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Add_7?
$convLSTM_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_1/clip_by_value_2/Minimum/y?
"convLSTM_1/clip_by_value_2/MinimumMinimumconvLSTM_1/Add_7:z:0-convLSTM_1/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2$
"convLSTM_1/clip_by_value_2/Minimum?
convLSTM_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_2/y?
convLSTM_1/clip_by_value_2Maximum&convLSTM_1/clip_by_value_2/Minimum:z:0%convLSTM_1/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/clip_by_value_2
convLSTM_1/Tanh_1TanhconvLSTM_1/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/Tanh_1?
convLSTM_1/mul_5MulconvLSTM_1/clip_by_value_2:z:0convLSTM_1/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
convLSTM_1/mul_5?
(convLSTM_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2*
(convLSTM_1/TensorArrayV2_1/element_shape?
convLSTM_1/TensorArrayV2_1TensorListReserve1convLSTM_1/TensorArrayV2_1/element_shape:output:0!convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_1/TensorArrayV2_1d
convLSTM_1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_1/time?
#convLSTM_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#convLSTM_1/while/maximum_iterations?
convLSTM_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_1/while/loop_counter?
convLSTM_1/whileWhile&convLSTM_1/while/loop_counter:output:0,convLSTM_1/while/maximum_iterations:output:0convLSTM_1/time:output:0#convLSTM_1/TensorArrayV2_1:handle:0convLSTM_1/convolution:output:0convLSTM_1/convolution:output:0!convLSTM_1/strided_slice:output:0BconvLSTM_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(convlstm_1_split_readvariableop_resource*convlstm_1_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *$
_read_only_resource_inputs
	*(
body R
convLSTM_1_while_body_605070*(
cond R
convLSTM_1_while_cond_605069*[
output_shapesJ
H: : : : :?????????v?:?????????v?: : : : *
parallel_iterations 2
convLSTM_1/while?
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2=
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape?
-convLSTM_1/TensorArrayV2Stack/TensorListStackTensorListStackconvLSTM_1/while:output:3DconvLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????v?*
element_dtype02/
-convLSTM_1/TensorArrayV2Stack/TensorListStack?
 convLSTM_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 convLSTM_1/strided_slice_2/stack?
"convLSTM_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"convLSTM_1/strided_slice_2/stack_1?
"convLSTM_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_2/stack_2?
convLSTM_1/strided_slice_2StridedSlice6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0)convLSTM_1/strided_slice_2/stack:output:0+convLSTM_1/strided_slice_2/stack_1:output:0+convLSTM_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:?????????v?*
shrink_axis_mask2
convLSTM_1/strided_slice_2?
convLSTM_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose_1/perm?
convLSTM_1/transpose_1	Transpose6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0$convLSTM_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :?????????v?2
convLSTM_1/transpose_1?
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2 
time_distributed/Reshape/shape?
time_distributed/ReshapeReshapeconvLSTM_1/transpose_1:y:0'time_distributed/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape?
&time_distributed/max_pooling2d/MaxPoolMaxPool!time_distributed/Reshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2(
&time_distributed/max_pooling2d/MaxPool?
 time_distributed/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2"
 time_distributed/Reshape_1/shape?
time_distributed/Reshape_1Reshape/time_distributed/max_pooling2d/MaxPool:output:0)time_distributed/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
time_distributed/Reshape_1?
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2"
 time_distributed/Reshape_2/shape?
time_distributed/Reshape_2ReshapeconvLSTM_1/transpose_1:y:0)time_distributed/Reshape_2/shape:output:0*
T0*0
_output_shapes
:?????????v?2
time_distributed/Reshape_2?
 time_distributed_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2"
 time_distributed_1/Reshape/shape?
time_distributed_1/ReshapeReshape#time_distributed/Reshape_1:output:0)time_distributed_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape?
>time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOpGtime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp?
@time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpItime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1?
/time_distributed_1/batchnorm_1/FusedBatchNormV3FusedBatchNormV3#time_distributed_1/Reshape:output:0$time_distributed_1_batchnorm_1_scale%time_distributed_1_batchnorm_1_offsetFtime_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0Htime_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????;O:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/time_distributed_1/batchnorm_1/FusedBatchNormV3?
-time_distributed_1/batchnorm_1/AssignNewValueAssignVariableOpGtime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_resource<time_distributed_1/batchnorm_1/FusedBatchNormV3:batch_mean:0?^time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp*Z
_classP
NLloc:@time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02/
-time_distributed_1/batchnorm_1/AssignNewValue?
/time_distributed_1/batchnorm_1/AssignNewValue_1AssignVariableOpItime_distributed_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource@time_distributed_1/batchnorm_1/FusedBatchNormV3:batch_variance:0A^time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*\
_classR
PNloc:@time_distributed_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype021
/time_distributed_1/batchnorm_1/AssignNewValue_1?
"time_distributed_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2$
"time_distributed_1/Reshape_1/shape?
time_distributed_1/Reshape_1Reshape3time_distributed_1/batchnorm_1/FusedBatchNormV3:y:0+time_distributed_1/Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
time_distributed_1/Reshape_1?
"time_distributed_1/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2$
"time_distributed_1/Reshape_2/shape?
time_distributed_1/Reshape_2Reshape#time_distributed/Reshape_1:output:0+time_distributed_1/Reshape_2/shape:output:0*
T0*/
_output_shapes
:?????????;O2
time_distributed_1/Reshape_2?
convLSTM_2/zeros_like	ZerosLike%time_distributed_1/Reshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2
convLSTM_2/zeros_like?
 convLSTM_2/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_2/Sum/reduction_indices?
convLSTM_2/SumSumconvLSTM_2/zeros_like:y:0)convLSTM_2/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
convLSTM_2/Sum?
 convLSTM_2/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 convLSTM_2/zeros/shape_as_tensoru
convLSTM_2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/zeros/Const?
convLSTM_2/zerosFill)convLSTM_2/zeros/shape_as_tensor:output:0convLSTM_2/zeros/Const:output:0*
T0*&
_output_shapes
: 2
convLSTM_2/zeros?
convLSTM_2/convolutionConv2DconvLSTM_2/Sum:output:0convLSTM_2/zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution?
convLSTM_2/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_2/transpose/perm?
convLSTM_2/transpose	Transpose%time_distributed_1/Reshape_1:output:0"convLSTM_2/transpose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2
convLSTM_2/transposel
convLSTM_2/ShapeShapeconvLSTM_2/transpose:y:0*
T0*
_output_shapes
:2
convLSTM_2/Shape?
convLSTM_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
convLSTM_2/strided_slice/stack?
 convLSTM_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_2/strided_slice/stack_1?
 convLSTM_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_2/strided_slice/stack_2?
convLSTM_2/strided_sliceStridedSliceconvLSTM_2/Shape:output:0'convLSTM_2/strided_slice/stack:output:0)convLSTM_2/strided_slice/stack_1:output:0)convLSTM_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
convLSTM_2/strided_slice?
&convLSTM_2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&convLSTM_2/TensorArrayV2/element_shape?
convLSTM_2/TensorArrayV2TensorListReserve/convLSTM_2/TensorArrayV2/element_shape:output:0!convLSTM_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_2/TensorArrayV2?
@convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      2B
@convLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shape?
2convLSTM_2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconvLSTM_2/transpose:y:0IconvLSTM_2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2convLSTM_2/TensorArrayUnstack/TensorListFromTensor?
 convLSTM_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 convLSTM_2/strided_slice_1/stack?
"convLSTM_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_2/strided_slice_1/stack_1?
"convLSTM_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_2/strided_slice_1/stack_2?
convLSTM_2/strided_slice_1StridedSliceconvLSTM_2/transpose:y:0)convLSTM_2/strided_slice_1/stack:output:0+convLSTM_2/strided_slice_1/stack_1:output:0+convLSTM_2/strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
convLSTM_2/strided_slice_1f
convLSTM_2/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/Constz
convLSTM_2/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/split/split_dim?
convLSTM_2/split/ReadVariableOpReadVariableOp(convlstm_2_split_readvariableop_resource*'
_output_shapes
:?*
dtype02!
convLSTM_2/split/ReadVariableOp?
convLSTM_2/splitSplit#convLSTM_2/split/split_dim:output:0'convLSTM_2/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
convLSTM_2/splitj
convLSTM_2/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/Const_1~
convLSTM_2/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_2/split_1/split_dim?
!convLSTM_2/split_1/ReadVariableOpReadVariableOp*convlstm_2_split_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02#
!convLSTM_2/split_1/ReadVariableOp?
convLSTM_2/split_1Split%convLSTM_2/split_1/split_dim:output:0)convLSTM_2/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
convLSTM_2/split_1?
convLSTM_2/convolution_1Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_1?
convLSTM_2/convolution_2Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_2?
convLSTM_2/convolution_3Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_3?
convLSTM_2/convolution_4Conv2D#convLSTM_2/strided_slice_1:output:0convLSTM_2/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convLSTM_2/convolution_4?
convLSTM_2/convolution_5Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_5?
convLSTM_2/convolution_6Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_6?
convLSTM_2/convolution_7Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_7?
convLSTM_2/convolution_8Conv2DconvLSTM_2/convolution:output:0convLSTM_2/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convLSTM_2/convolution_8?
convLSTM_2/addAddV2!convLSTM_2/convolution_1:output:0!convLSTM_2/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/addm
convLSTM_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/Const_2m
convLSTM_2/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/Const_3?
convLSTM_2/MulMulconvLSTM_2/add:z:0convLSTM_2/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Mul?
convLSTM_2/Add_1AddconvLSTM_2/Mul:z:0convLSTM_2/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Add_1?
"convLSTM_2/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"convLSTM_2/clip_by_value/Minimum/y?
 convLSTM_2/clip_by_value/MinimumMinimumconvLSTM_2/Add_1:z:0+convLSTM_2/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2"
 convLSTM_2/clip_by_value/Minimum}
convLSTM_2/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/clip_by_value/y?
convLSTM_2/clip_by_valueMaximum$convLSTM_2/clip_by_value/Minimum:z:0#convLSTM_2/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/clip_by_value?
convLSTM_2/add_2AddV2!convLSTM_2/convolution_2:output:0!convLSTM_2/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_2m
convLSTM_2/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/Const_4m
convLSTM_2/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/Const_5?
convLSTM_2/Mul_1MulconvLSTM_2/add_2:z:0convLSTM_2/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Mul_1?
convLSTM_2/Add_3AddconvLSTM_2/Mul_1:z:0convLSTM_2/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Add_3?
$convLSTM_2/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_2/clip_by_value_1/Minimum/y?
"convLSTM_2/clip_by_value_1/MinimumMinimumconvLSTM_2/Add_3:z:0-convLSTM_2/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2$
"convLSTM_2/clip_by_value_1/Minimum?
convLSTM_2/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/clip_by_value_1/y?
convLSTM_2/clip_by_value_1Maximum&convLSTM_2/clip_by_value_1/Minimum:z:0%convLSTM_2/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/clip_by_value_1?
convLSTM_2/mul_2MulconvLSTM_2/clip_by_value_1:z:0convLSTM_2/convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/mul_2?
convLSTM_2/add_4AddV2!convLSTM_2/convolution_3:output:0!convLSTM_2/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_4z
convLSTM_2/TanhTanhconvLSTM_2/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Tanh?
convLSTM_2/mul_3MulconvLSTM_2/clip_by_value:z:0convLSTM_2/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/mul_3?
convLSTM_2/add_5AddV2convLSTM_2/mul_2:z:0convLSTM_2/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_5?
convLSTM_2/add_6AddV2!convLSTM_2/convolution_4:output:0!convLSTM_2/convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/add_6m
convLSTM_2/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
convLSTM_2/Const_6m
convLSTM_2/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_2/Const_7?
convLSTM_2/Mul_4MulconvLSTM_2/add_6:z:0convLSTM_2/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Mul_4?
convLSTM_2/Add_7AddconvLSTM_2/Mul_4:z:0convLSTM_2/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Add_7?
$convLSTM_2/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$convLSTM_2/clip_by_value_2/Minimum/y?
"convLSTM_2/clip_by_value_2/MinimumMinimumconvLSTM_2/Add_7:z:0-convLSTM_2/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2$
"convLSTM_2/clip_by_value_2/Minimum?
convLSTM_2/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_2/clip_by_value_2/y?
convLSTM_2/clip_by_value_2Maximum&convLSTM_2/clip_by_value_2/Minimum:z:0%convLSTM_2/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/clip_by_value_2~
convLSTM_2/Tanh_1TanhconvLSTM_2/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/Tanh_1?
convLSTM_2/mul_5MulconvLSTM_2/clip_by_value_2:z:0convLSTM_2/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
convLSTM_2/mul_5?
(convLSTM_2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2*
(convLSTM_2/TensorArrayV2_1/element_shape?
convLSTM_2/TensorArrayV2_1TensorListReserve1convLSTM_2/TensorArrayV2_1/element_shape:output:0!convLSTM_2/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_2/TensorArrayV2_1d
convLSTM_2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_2/time?
#convLSTM_2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#convLSTM_2/while/maximum_iterations?
convLSTM_2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_2/while/loop_counter?
convLSTM_2/whileWhile&convLSTM_2/while/loop_counter:output:0,convLSTM_2/while/maximum_iterations:output:0convLSTM_2/time:output:0#convLSTM_2/TensorArrayV2_1:handle:0convLSTM_2/convolution:output:0convLSTM_2/convolution:output:0!convLSTM_2/strided_slice:output:0BconvLSTM_2/TensorArrayUnstack/TensorListFromTensor:output_handle:0(convlstm_2_split_readvariableop_resource*convlstm_2_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*(
body R
convLSTM_2_while_body_605295*(
cond R
convLSTM_2_while_cond_605294*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
convLSTM_2/while?
;convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2=
;convLSTM_2/TensorArrayV2Stack/TensorListStack/element_shape?
-convLSTM_2/TensorArrayV2Stack/TensorListStackTensorListStackconvLSTM_2/while:output:3DconvLSTM_2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02/
-convLSTM_2/TensorArrayV2Stack/TensorListStack?
 convLSTM_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 convLSTM_2/strided_slice_2/stack?
"convLSTM_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"convLSTM_2/strided_slice_2/stack_1?
"convLSTM_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_2/strided_slice_2/stack_2?
convLSTM_2/strided_slice_2StridedSlice6convLSTM_2/TensorArrayV2Stack/TensorListStack:tensor:0)convLSTM_2/strided_slice_2/stack:output:0+convLSTM_2/strided_slice_2/stack_1:output:0+convLSTM_2/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
convLSTM_2/strided_slice_2?
convLSTM_2/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_2/transpose_1/perm?
convLSTM_2/transpose_1	Transpose6convLSTM_2/TensorArrayV2Stack/TensorListStack:tensor:0$convLSTM_2/transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2
convLSTM_2/transpose_1?
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indices?
global_max_pooling2d/MaxMax#convLSTM_2/strided_slice_2:output:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:????????? 2
global_max_pooling2d/Max?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul!global_max_pooling2d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const?
IdentityIdentitydense/Sigmoid:y:0^convLSTM_1/while^convLSTM_2/while.^time_distributed_1/batchnorm_1/AssignNewValue0^time_distributed_1/batchnorm_1/AssignNewValue_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????x?::::::::::2$
convLSTM_1/whileconvLSTM_1/while2$
convLSTM_2/whileconvLSTM_2/while2^
-time_distributed_1/batchnorm_1/AssignNewValue-time_distributed_1/batchnorm_1/AssignNewValue2b
/time_distributed_1/batchnorm_1/AssignNewValue_1/time_distributed_1/batchnorm_1/AssignNewValue_1:\ X
4
_output_shapes"
 :?????????x?
 
_user_specified_nameinputs
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_602896

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicew
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/PartitionedCallPartitionedCallReshape:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_6028092
max_pooling2d/PartitionedCallq
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape_1/shape/0h
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :;2
Reshape_1/shape/2h
Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :O2
Reshape_1/shape/3h
Reshape_1/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/4?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0Reshape_1/shape/3:output:0Reshape_1/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1Reshape&max_pooling2d/PartitionedCall:output:0Reshape_1/shape:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2
	Reshape_1{
IdentityIdentityReshape_1:output:0*
T0*<
_output_shapes*
(:&??????????????????;O2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'??????????????????v?:e a
=
_output_shapes+
):'??????????????????v?
 
_user_specified_nameinputs
?
?
+__inference_convLSTM_1_layer_call_fn_606317
inputs_0
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'??????????????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_6026882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*=
_output_shapes+
):'??????????????????v?2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'??????????????????x?::22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'??????????????????x?
"
_user_specified_name
inputs/0
?;
?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_603347

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstatessplit_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstatessplit_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstatessplit_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstatessplit_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7{
addAddV2convolution:output:0convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1n
mul_2Mulclip_by_value_1:z:0states_1*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Conste
IdentityIdentity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identityi

Identity_1Identity	mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_1i

Identity_2Identity	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*l
_input_shapes[
Y:?????????;O:?????????9M :?????????9M :::W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????9M 
 
_user_specified_namestates:WS
/
_output_shapes
:?????????9M 
 
_user_specified_namestates
?	
?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608047

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+???????????????????????????:::::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?X
?
while_body_607265
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*'
_output_shapes
:?*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*'
_output_shapes
: ?*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3~
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*/
_output_shapes
:?????????9M 2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_4k

while/TanhTanhwhile/add_4:z:0*
T0*/
_output_shapes
:?????????9M 2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
while/clip_by_value_2o
while/Tanh_1Tanhwhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3{
while/Identity_4Identitywhile/mul_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4{
while/Identity_5Identitywhile/add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
??
?

)functional_1_convLSTM_1_while_body_601874L
Hfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_loop_counterR
Nfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_maximum_iterations-
)functional_1_convlstm_1_while_placeholder/
+functional_1_convlstm_1_while_placeholder_1/
+functional_1_convlstm_1_while_placeholder_2/
+functional_1_convlstm_1_while_placeholder_3I
Efunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_slice_0?
?functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0A
=functional_1_convlstm_1_while_split_readvariableop_resource_0C
?functional_1_convlstm_1_while_split_1_readvariableop_resource_0*
&functional_1_convlstm_1_while_identity,
(functional_1_convlstm_1_while_identity_1,
(functional_1_convlstm_1_while_identity_2,
(functional_1_convlstm_1_while_identity_3,
(functional_1_convlstm_1_while_identity_4,
(functional_1_convlstm_1_while_identity_5G
Cfunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_slice?
?functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor?
;functional_1_convlstm_1_while_split_readvariableop_resourceA
=functional_1_convlstm_1_while_split_1_readvariableop_resource??
Ofunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      2Q
Ofunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
Afunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0)functional_1_convlstm_1_while_placeholderXfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02C
Afunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem?
#functional_1/convLSTM_1/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#functional_1/convLSTM_1/while/Const?
-functional_1/convLSTM_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/convLSTM_1/while/split/split_dim?
2functional_1/convLSTM_1/while/split/ReadVariableOpReadVariableOp=functional_1_convlstm_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype024
2functional_1/convLSTM_1/while/split/ReadVariableOp?
#functional_1/convLSTM_1/while/splitSplit6functional_1/convLSTM_1/while/split/split_dim:output:0:functional_1/convLSTM_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2%
#functional_1/convLSTM_1/while/split?
%functional_1/convLSTM_1/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_1/while/Const_1?
/functional_1/convLSTM_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/convLSTM_1/while/split_1/split_dim?
4functional_1/convLSTM_1/while/split_1/ReadVariableOpReadVariableOp?functional_1_convlstm_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype026
4functional_1/convLSTM_1/while/split_1/ReadVariableOp?
%functional_1/convLSTM_1/while/split_1Split8functional_1/convLSTM_1/while/split_1/split_dim:output:0<functional_1/convLSTM_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2'
%functional_1/convLSTM_1/while/split_1?
)functional_1/convLSTM_1/while/convolutionConv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2+
)functional_1/convLSTM_1/while/convolution?
+functional_1/convLSTM_1/while/convolution_1Conv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2-
+functional_1/convLSTM_1/while/convolution_1?
+functional_1/convLSTM_1/while/convolution_2Conv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2-
+functional_1/convLSTM_1/while/convolution_2?
+functional_1/convLSTM_1/while/convolution_3Conv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2-
+functional_1/convLSTM_1/while/convolution_3?
+functional_1/convLSTM_1/while/convolution_4Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_4?
+functional_1/convLSTM_1/while/convolution_5Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_5?
+functional_1/convLSTM_1/while/convolution_6Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_6?
+functional_1/convLSTM_1/while/convolution_7Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_7?
!functional_1/convLSTM_1/while/addAddV22functional_1/convLSTM_1/while/convolution:output:04functional_1/convLSTM_1/while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2#
!functional_1/convLSTM_1/while/add?
%functional_1/convLSTM_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%functional_1/convLSTM_1/while/Const_2?
%functional_1/convLSTM_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_1/while/Const_3?
!functional_1/convLSTM_1/while/MulMul%functional_1/convLSTM_1/while/add:z:0.functional_1/convLSTM_1/while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2#
!functional_1/convLSTM_1/while/Mul?
#functional_1/convLSTM_1/while/Add_1Add%functional_1/convLSTM_1/while/Mul:z:0.functional_1/convLSTM_1/while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/Add_1?
5functional_1/convLSTM_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5functional_1/convLSTM_1/while/clip_by_value/Minimum/y?
3functional_1/convLSTM_1/while/clip_by_value/MinimumMinimum'functional_1/convLSTM_1/while/Add_1:z:0>functional_1/convLSTM_1/while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?25
3functional_1/convLSTM_1/while/clip_by_value/Minimum?
-functional_1/convLSTM_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-functional_1/convLSTM_1/while/clip_by_value/y?
+functional_1/convLSTM_1/while/clip_by_valueMaximum7functional_1/convLSTM_1/while/clip_by_value/Minimum:z:06functional_1/convLSTM_1/while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2-
+functional_1/convLSTM_1/while/clip_by_value?
#functional_1/convLSTM_1/while/add_2AddV24functional_1/convLSTM_1/while/convolution_1:output:04functional_1/convLSTM_1/while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/add_2?
%functional_1/convLSTM_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%functional_1/convLSTM_1/while/Const_4?
%functional_1/convLSTM_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_1/while/Const_5?
#functional_1/convLSTM_1/while/Mul_1Mul'functional_1/convLSTM_1/while/add_2:z:0.functional_1/convLSTM_1/while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/Mul_1?
#functional_1/convLSTM_1/while/Add_3Add'functional_1/convLSTM_1/while/Mul_1:z:0.functional_1/convLSTM_1/while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/Add_3?
7functional_1/convLSTM_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7functional_1/convLSTM_1/while/clip_by_value_1/Minimum/y?
5functional_1/convLSTM_1/while/clip_by_value_1/MinimumMinimum'functional_1/convLSTM_1/while/Add_3:z:0@functional_1/convLSTM_1/while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?27
5functional_1/convLSTM_1/while/clip_by_value_1/Minimum?
/functional_1/convLSTM_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/functional_1/convLSTM_1/while/clip_by_value_1/y?
-functional_1/convLSTM_1/while/clip_by_value_1Maximum9functional_1/convLSTM_1/while/clip_by_value_1/Minimum:z:08functional_1/convLSTM_1/while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2/
-functional_1/convLSTM_1/while/clip_by_value_1?
#functional_1/convLSTM_1/while/mul_2Mul1functional_1/convLSTM_1/while/clip_by_value_1:z:0+functional_1_convlstm_1_while_placeholder_3*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/mul_2?
#functional_1/convLSTM_1/while/add_4AddV24functional_1/convLSTM_1/while/convolution_2:output:04functional_1/convLSTM_1/while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/add_4?
"functional_1/convLSTM_1/while/TanhTanh'functional_1/convLSTM_1/while/add_4:z:0*
T0*0
_output_shapes
:?????????v?2$
"functional_1/convLSTM_1/while/Tanh?
#functional_1/convLSTM_1/while/mul_3Mul/functional_1/convLSTM_1/while/clip_by_value:z:0&functional_1/convLSTM_1/while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/mul_3?
#functional_1/convLSTM_1/while/add_5AddV2'functional_1/convLSTM_1/while/mul_2:z:0'functional_1/convLSTM_1/while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/add_5?
#functional_1/convLSTM_1/while/add_6AddV24functional_1/convLSTM_1/while/convolution_3:output:04functional_1/convLSTM_1/while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/add_6?
%functional_1/convLSTM_1/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%functional_1/convLSTM_1/while/Const_6?
%functional_1/convLSTM_1/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_1/while/Const_7?
#functional_1/convLSTM_1/while/Mul_4Mul'functional_1/convLSTM_1/while/add_6:z:0.functional_1/convLSTM_1/while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/Mul_4?
#functional_1/convLSTM_1/while/Add_7Add'functional_1/convLSTM_1/while/Mul_4:z:0.functional_1/convLSTM_1/while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/Add_7?
7functional_1/convLSTM_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??29
7functional_1/convLSTM_1/while/clip_by_value_2/Minimum/y?
5functional_1/convLSTM_1/while/clip_by_value_2/MinimumMinimum'functional_1/convLSTM_1/while/Add_7:z:0@functional_1/convLSTM_1/while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?27
5functional_1/convLSTM_1/while/clip_by_value_2/Minimum?
/functional_1/convLSTM_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/functional_1/convLSTM_1/while/clip_by_value_2/y?
-functional_1/convLSTM_1/while/clip_by_value_2Maximum9functional_1/convLSTM_1/while/clip_by_value_2/Minimum:z:08functional_1/convLSTM_1/while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2/
-functional_1/convLSTM_1/while/clip_by_value_2?
$functional_1/convLSTM_1/while/Tanh_1Tanh'functional_1/convLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:?????????v?2&
$functional_1/convLSTM_1/while/Tanh_1?
#functional_1/convLSTM_1/while/mul_5Mul1functional_1/convLSTM_1/while/clip_by_value_2:z:0(functional_1/convLSTM_1/while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2%
#functional_1/convLSTM_1/while/mul_5?
Bfunctional_1/convLSTM_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+functional_1_convlstm_1_while_placeholder_1)functional_1_convlstm_1_while_placeholder'functional_1/convLSTM_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype02D
Bfunctional_1/convLSTM_1/while/TensorArrayV2Write/TensorListSetItem?
%functional_1/convLSTM_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_1/while/add_8/y?
#functional_1/convLSTM_1/while/add_8AddV2)functional_1_convlstm_1_while_placeholder.functional_1/convLSTM_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/convLSTM_1/while/add_8?
%functional_1/convLSTM_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_1/while/add_9/y?
#functional_1/convLSTM_1/while/add_9AddV2Hfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_loop_counter.functional_1/convLSTM_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/convLSTM_1/while/add_9?
&functional_1/convLSTM_1/while/IdentityIdentity'functional_1/convLSTM_1/while/add_9:z:0*
T0*
_output_shapes
: 2(
&functional_1/convLSTM_1/while/Identity?
(functional_1/convLSTM_1/while/Identity_1IdentityNfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_maximum_iterations*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_1/while/Identity_1?
(functional_1/convLSTM_1/while/Identity_2Identity'functional_1/convLSTM_1/while/add_8:z:0*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_1/while/Identity_2?
(functional_1/convLSTM_1/while/Identity_3IdentityRfunctional_1/convLSTM_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_1/while/Identity_3?
(functional_1/convLSTM_1/while/Identity_4Identity'functional_1/convLSTM_1/while/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2*
(functional_1/convLSTM_1/while/Identity_4?
(functional_1/convLSTM_1/while/Identity_5Identity'functional_1/convLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:?????????v?2*
(functional_1/convLSTM_1/while/Identity_5"?
Cfunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_sliceEfunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_slice_0"Y
&functional_1_convlstm_1_while_identity/functional_1/convLSTM_1/while/Identity:output:0"]
(functional_1_convlstm_1_while_identity_11functional_1/convLSTM_1/while/Identity_1:output:0"]
(functional_1_convlstm_1_while_identity_21functional_1/convLSTM_1/while/Identity_2:output:0"]
(functional_1_convlstm_1_while_identity_31functional_1/convLSTM_1/while/Identity_3:output:0"]
(functional_1_convlstm_1_while_identity_41functional_1/convLSTM_1/while/Identity_4:output:0"]
(functional_1_convlstm_1_while_identity_51functional_1/convLSTM_1/while/Identity_5:output:0"?
=functional_1_convlstm_1_while_split_1_readvariableop_resource?functional_1_convlstm_1_while_split_1_readvariableop_resource_0"|
;functional_1_convlstm_1_while_split_readvariableop_resource=functional_1_convlstm_1_while_split_readvariableop_resource_0"?
?functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor?functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?X
?
while_body_606612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0)
%while_split_readvariableop_resource_0+
'while_split_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor'
#while_split_readvariableop_resource)
%while_split_1_readvariableop_resource??
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????x   ?      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:?????????x?*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem\
while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/Constp
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split/split_dim?
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOp?
while/splitSplitwhile/split/split_dim:output:0"while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split`
while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/Const_1t
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
while/split_1/split_dim?
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOp?
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1?
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution?
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_1?
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_2?
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
while/convolution_3?
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_4?
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_5?
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_6?
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
while/convolution_7?
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_2c
while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_3
	while/MulMulwhile/add:z:0while/Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
	while/Mul?
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_1?
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/clip_by_value/Minimum/y?
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y?
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value?
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5?
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_1?
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_3?
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_1/Minimum/y?
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y?
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_1?
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:?????????v?2
while/mul_2?
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:?????????v?2

while/Tanh?
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_3?
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
while/add_5?
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7?
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
while/Mul_4?
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
while/Add_7?
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
while/clip_by_value_2/Minimum/y?
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y?
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Tanh_1?
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
while/mul_5?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem`
while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_8/yo
while/add_8AddV2while_placeholderwhile/add_8/y:output:0*
T0*
_output_shapes
: 2
while/add_8`
while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_9/yv
while/add_9AddV2while_while_loop_counterwhile/add_9/y:output:0*
T0*
_output_shapes
: 2
while/add_9^
while/IdentityIdentitywhile/add_9:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1b
while/Identity_2Identitywhile/add_8:z:0*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:?????????v?2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :?????????v?:?????????v?: : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :62
0
_output_shapes
:?????????v?:62
0
_output_shapes
:?????????v?:

_output_shapes
: :

_output_shapes
: 
?"
?
while_body_603595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_603619_0
while_603621_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_603619
while_603621??while/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*/
_output_shapes
:?????????;O*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_603619_0while_603621_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *e
_output_shapesS
Q:?????????9M :?????????9M :?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_6032802
while/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder&while/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1~
while/IdentityIdentitywhile/add_1:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2
while/Identity_4?
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2
while/Identity_5"
while_603619while_603619_0"
while_603621while_603621_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*]
_input_shapesL
J: : : : :?????????9M :?????????9M : : ::2>
while/StatefulPartitionedCallwhile/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :51
/
_output_shapes
:?????????9M :51
/
_output_shapes
:?????????9M :

_output_shapes
: :

_output_shapes
: 
?g
?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_604531

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity??whilek

zeros_like	ZerosLikeinputs*
T0*3
_output_shapes!
:?????????;O2

zeros_likep
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices{
SumSumzeros_like:y:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:?????????;O2
Sum?
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Const}
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*&
_output_shapes
: 2
zeros?
convolutionConv2DSum:output:0zeros:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*3
_output_shapes!
:?????????;O2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????;   O      27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????;O*
shrink_axis_mask2
strided_slice_1P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*'
_output_shapes
:?*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H: : : : *
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*'
_output_shapes
: ?*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H:  :  :  :  *
	num_split2	
split_1?
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*/
_output_shapes
:?????????9M *
paddingVALID*
strides
2
convolution_4?
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_7?
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*/
_output_shapes
:?????????9M *
paddingSAME*
strides
2
convolution_8}
addAddV2convolution_1:output:0convolution_5:output:0*
T0*/
_output_shapes
:?????????9M 2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3f
MulMuladd:z:0Const_2:output:0*
T0*/
_output_shapes
:?????????9M 2
Mulj
Add_1AddMul:z:0Const_3:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value?
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*/
_output_shapes
:?????????9M 2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5l
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_1l
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_1z
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*/
_output_shapes
:?????????9M 2
mul_2?
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*/
_output_shapes
:?????????9M 2
add_4Y
TanhTanh	add_4:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanhl
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_3g
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*/
_output_shapes
:?????????9M 2
add_5?
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*/
_output_shapes
:?????????9M 2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7l
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*/
_output_shapes
:?????????9M 2
Mul_4l
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*/
_output_shapes
:?????????9M 2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*/
_output_shapes
:?????????9M 2
clip_by_value_2]
Tanh_1Tanh	add_5:z:0*
T0*/
_output_shapes
:?????????9M 2
Tanh_1p
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*/
_output_shapes
:?????????9M 2
mul_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*Z
_output_shapesH
F: : : : :?????????9M :?????????9M : : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_604415*
condR
while_cond_604414*Y
output_shapesH
F: : : : :?????????9M :?????????9M : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"????9   M       22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:?????????9M *
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????9M *
shrink_axis_mask2
strided_slice_2?
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*3
_output_shapes!
:?????????9M 2
transpose_1?
#convLSTM_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_2/kernel/Regularizer/Const|
IdentityIdentitystrided_slice_2:output:0^while*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????;O::2
whilewhile:[ W
3
_output_shapes!
:?????????;O
 
_user_specified_nameinputs
?
?
,__inference_batchnorm_1_layer_call_fn_608120

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????;O* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_6030232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????;O2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????;O::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs
?
?
,__inference_batchnorm_1_layer_call_fn_608133

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????;O*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_6030392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????;O2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:?????????;O::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????;O
 
_user_specified_nameinputs
?
h
L__inference_time_distributed_layer_call_and_return_conditional_losses_606755

inputs
identityw
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"????v   ?      2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????v?2	
Reshape?
max_pooling2d/MaxPoolMaxPoolReshape:output:0*/
_output_shapes
:?????????;O*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*)
value B"????   ;   O      2
Reshape_1/shape?
	Reshape_1Reshapemax_pooling2d/MaxPool:output:0Reshape_1/shape:output:0*
T0*3
_output_shapes!
:?????????;O2
	Reshape_1r
IdentityIdentityReshape_1:output:0*
T0*3
_output_shapes!
:?????????;O2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :?????????v?:\ X
4
_output_shapes"
 :?????????v?
 
_user_specified_nameinputs
?
?
2__inference_conv_lst_m2d_cell_layer_call_fn_608008

inputs
states_0
states_1
unknown
	unknown_0
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:?????????v?:?????????v?:?????????v?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_6023792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????v?2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:?????????x?:?????????v?:?????????v?::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????x?
 
_user_specified_nameinputs:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/0:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/1
?
?
+__inference_convLSTM_2_layer_call_fn_607390
inputs_0
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????9M *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_6036602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????9M 2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:&??????????????????;O::22
StatefulPartitionedCallStatefulPartitionedCall:f b
<
_output_shapes*
(:&??????????????????;O
"
_user_specified_name
inputs/0
?;
?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_607911

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2?P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOp?
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
splitT
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split_1/split_dim?
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOp?
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1?
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution?
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_1?
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_2?
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:?????????v?*
paddingVALID*
strides
2
convolution_3?
convolution_4Conv2Dstates_0split_1:output:0*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_4?
convolution_5Conv2Dstates_0split_1:output:1*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_5?
convolution_6Conv2Dstates_0split_1:output:2*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_6?
convolution_7Conv2Dstates_0split_1:output:3*
T0*0
_output_shapes
:?????????v?*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:?????????v?2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3g
MulMuladd:z:0Const_2:output:0*
T0*0
_output_shapes
:?????????v?2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:?????????v?2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value?
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:?????????v?2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5m
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:?????????v?2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:?????????v?2
mul_2?
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:?????????v?2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:?????????v?2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:?????????v?2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????v?2
add_5?
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:?????????v?2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7m
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*0
_output_shapes
:?????????v?2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:?????????v?2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_2/Minimum/y?
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y?
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:?????????v?2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:?????????v?2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:?????????v?2
mul_5?
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constf
IdentityIdentity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:?????????v?2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:?????????x?:?????????v?:?????????v?:::X T
0
_output_shapes
:?????????x?
 
_user_specified_nameinputs:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/0:ZV
0
_output_shapes
:?????????v?
"
_user_specified_name
states/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
H
input_1=
serving_default_input_1:0?????????x?9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?u
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?r
_tf_keras_network?r{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ConvLSTM2D", "config": {"name": "convLSTM_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "name": "convLSTM_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}}, "name": "time_distributed", "inbound_nodes": [[["convLSTM_1", 0, 0, {}]]]}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}}, "name": "time_distributed_1", "inbound_nodes": [[["time_distributed", 0, 0, {}]]]}, {"class_name": "ConvLSTM2D", "config": {"name": "convLSTM_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "name": "convLSTM_2", "inbound_nodes": [[["time_distributed_1", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "global_max_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling2d", "inbound_nodes": [[["convLSTM_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_max_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 120, 160, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ConvLSTM2D", "config": {"name": "convLSTM_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "name": "convLSTM_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}}, "name": "time_distributed", "inbound_nodes": [[["convLSTM_1", 0, 0, {}]]]}, {"class_name": "TimeDistributed", "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}}, "name": "time_distributed_1", "inbound_nodes": [[["time_distributed", 0, 0, {}]]]}, {"class_name": "ConvLSTM2D", "config": {"name": "convLSTM_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "name": "convLSTM_2", "inbound_nodes": [[["time_distributed_1", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "global_max_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling2d", "inbound_nodes": [[["convLSTM_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_max_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy_in_loss", "from_logits": false, "label_smoothing": 0}}, "metrics": [{"class_name": "BinaryCrossentropy", "config": {"name": "binary_crossentropy_in_metrics", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 7.999999797903001e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?
cell

state_spec
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_rnn_layer?{"class_name": "ConvLSTM2D", "name": "convLSTM_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "convLSTM_1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 120, 160, 4]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 120, 160, 4]}}
?
	layer
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TimeDistributed", "name": "time_distributed", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 118, 158, 16], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 118, 158, 16]}}
?

	layer
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TimeDistributed", "name": "time_distributed_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "time_distributed_1", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 59, 79, 16], "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 59, 79, 16]}}
?
cell

state_spec
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_rnn_layer?{"class_name": "ConvLSTM2D", "name": "convLSTM_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "convLSTM_2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 59, 79, 16]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 59, 79, 16]}}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling2D", "name": "global_max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
.iter

/beta_1

0beta_2
	1decay
2learning_rate(m?)m?3m?4m?7m?8m?(v?)v?3v?4v?7v?8v?"
	optimizer
X
30
41
52
63
74
85
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
J
30
41
72
83
(4
)5"
trackable_list_wrapper
?
		variables
9layer_metrics

regularization_losses
:metrics
;non_trainable_variables

<layers
=layer_regularization_losses
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?


3kernel
4recurrent_kernel
>	variables
?regularization_losses
@trainable_variables
A	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "ConvLSTM2DCell", "name": "conv_lst_m2d_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_lst_m2d_cell", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?

Bstates
	variables
Clayer_metrics
regularization_losses
Dmetrics
Enon_trainable_variables

Flayers
Glayer_regularization_losses
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
H	variables
Iregularization_losses
Jtrainable_variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Llayer_metrics
regularization_losses
Mmetrics
Nnon_trainable_variables

Olayers
Player_regularization_losses
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Qaxis
5moving_mean
6moving_variance
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batchnorm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}}
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Vlayer_metrics
regularization_losses
Wmetrics
Xnon_trainable_variables

Ylayers
Zlayer_regularization_losses
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


7kernel
8recurrent_kernel
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?{"class_name": "ConvLSTM2DCell", "name": "conv_lst_m2d_cell_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_lst_m2d_cell_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?

_states
 	variables
`layer_metrics
!regularization_losses
ametrics
bnon_trainable_variables

clayers
dlayer_regularization_losses
"trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
$	variables
elayer_metrics
%regularization_losses
fmetrics
gnon_trainable_variables

hlayers
ilayer_regularization_losses
&trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: 2dense/kernel
:2
dense/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
*	variables
jlayer_metrics
+regularization_losses
kmetrics
lnon_trainable_variables

mlayers
nlayer_regularization_losses
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)@2convLSTM_1/kernel
5:3@2convLSTM_1/recurrent_kernel
.:, (2time_distributed_1/moving_mean
2:0 (2"time_distributed_1/moving_variance
,:*?2convLSTM_2/kernel
6:4 ?2convLSTM_2/recurrent_kernel
 "
trackable_dict_wrapper
5
o0
p1
q2"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
>	variables
rlayer_metrics
?regularization_losses
smetrics
tnon_trainable_variables

ulayers
vlayer_regularization_losses
@trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!"
trackable_tuple_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
H	variables
wlayer_metrics
Iregularization_losses
xmetrics
ynon_trainable_variables

zlayers
{layer_regularization_losses
Jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
R	variables
|layer_metrics
Sregularization_losses
}metrics
~non_trainable_variables

layers
 ?layer_regularization_losses
Ttrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
[	variables
?layer_metrics
\regularization_losses
?metrics
?non_trainable_variables
?layers
 ?layer_regularization_losses
]trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!"
trackable_tuple_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "BinaryCrossentropy", "name": "binary_crossentropy_in_metrics", "dtype": "float32", "config": {"name": "binary_crossentropy_in_metrics", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}
?"
?true_positives
?true_negatives
?false_positives
?false_negatives
?	variables
?	keras_api"?!
_tf_keras_metric?!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:? (2true_positives
:? (2true_negatives
 :? (2false_positives
 :? (2false_negatives
@
?0
?1
?2
?3"
trackable_list_wrapper
.
?	variables"
_generic_user_object
#:! 2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.@2Adam/convLSTM_1/kernel/m
::8@2"Adam/convLSTM_1/recurrent_kernel/m
1:/?2Adam/convLSTM_2/kernel/m
;:9 ?2"Adam/convLSTM_2/recurrent_kernel/m
#:! 2Adam/dense/kernel/v
:2Adam/dense/bias/v
0:.@2Adam/convLSTM_1/kernel/v
::8@2"Adam/convLSTM_1/recurrent_kernel/v
1:/?2Adam/convLSTM_2/kernel/v
;:9 ?2"Adam/convLSTM_2/recurrent_kernel/v
?2?
!__inference__wrapped_model_602221?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+
input_1?????????x?
?2?
H__inference_functional_1_layer_call_and_return_conditional_losses_605855
H__inference_functional_1_layer_call_and_return_conditional_losses_605421
H__inference_functional_1_layer_call_and_return_conditional_losses_604789
H__inference_functional_1_layer_call_and_return_conditional_losses_604825?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_functional_1_layer_call_fn_604887
-__inference_functional_1_layer_call_fn_604948
-__inference_functional_1_layer_call_fn_605880
-__inference_functional_1_layer_call_fn_605905?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606308
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606728
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606107
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606527?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_convLSTM_1_layer_call_fn_606737
+__inference_convLSTM_1_layer_call_fn_606326
+__inference_convLSTM_1_layer_call_fn_606317
+__inference_convLSTM_1_layer_call_fn_606746?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_time_distributed_layer_call_and_return_conditional_losses_606764
L__inference_time_distributed_layer_call_and_return_conditional_losses_606792
L__inference_time_distributed_layer_call_and_return_conditional_losses_606810
L__inference_time_distributed_layer_call_and_return_conditional_losses_606755?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
1__inference_time_distributed_layer_call_fn_606820
1__inference_time_distributed_layer_call_fn_606815
1__inference_time_distributed_layer_call_fn_606769
1__inference_time_distributed_layer_call_fn_606774?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606880
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606948
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606851
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606928?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_time_distributed_1_layer_call_fn_606974
3__inference_time_distributed_1_layer_call_fn_606893
3__inference_time_distributed_1_layer_call_fn_606906
3__inference_time_distributed_1_layer_call_fn_606961?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607381
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607178
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607805
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607602?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_convLSTM_2_layer_call_fn_607390
+__inference_convLSTM_2_layer_call_fn_607814
+__inference_convLSTM_2_layer_call_fn_607823
+__inference_convLSTM_2_layer_call_fn_607399?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_603784?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
5__inference_global_max_pooling2d_layer_call_fn_603790?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_dense_layer_call_and_return_conditional_losses_607834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_607843?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3B1
$__inference_signature_wrapper_604985input_1
?2?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_607911
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_607978?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_conv_lst_m2d_cell_layer_call_fn_607993
2__inference_conv_lst_m2d_cell_layer_call_fn_608008?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_0_608013?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_602809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_layer_call_fn_602815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608031
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608047
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608091
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608107?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_batchnorm_1_layer_call_fn_608073
,__inference_batchnorm_1_layer_call_fn_608133
,__inference_batchnorm_1_layer_call_fn_608120
,__inference_batchnorm_1_layer_call_fn_608060?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_608201
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_608268?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_conv_lst_m2d_cell_1_layer_call_fn_608283
4__inference_conv_lst_m2d_cell_1_layer_call_fn_608298?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference_loss_fn_1_608303?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1?
!__inference__wrapped_model_602221|34??5678()=?:
3?0
.?+
input_1?????????x?
? "-?*
(
dense?
dense??????????
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608031???56M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608047???56M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608091t??56;?8
1?.
(?%
inputs?????????;O
p
? "-?*
#? 
0?????????;O
? ?
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_608107t??56;?8
1?.
(?%
inputs?????????;O
p 
? "-?*
#? 
0?????????;O
? ?
,__inference_batchnorm_1_layer_call_fn_608060???56M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
,__inference_batchnorm_1_layer_call_fn_608073???56M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
,__inference_batchnorm_1_layer_call_fn_608120g??56;?8
1?.
(?%
inputs?????????;O
p
? " ??????????;O?
,__inference_batchnorm_1_layer_call_fn_608133g??56;?8
1?.
(?%
inputs?????????;O
p 
? " ??????????;O?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606107?34X?U
N?K
=?:
8?5
inputs/0'??????????????????x?

 
p

 
? ";?8
1?.
0'??????????????????v?
? ?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606308?34X?U
N?K
=?:
8?5
inputs/0'??????????????????x?

 
p 

 
? ";?8
1?.
0'??????????????????v?
? ?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606527?34H?E
>?;
-?*
inputs?????????x?

 
p

 
? "2?/
(?%
0?????????v?
? ?
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_606728?34H?E
>?;
-?*
inputs?????????x?

 
p 

 
? "2?/
(?%
0?????????v?
? ?
+__inference_convLSTM_1_layer_call_fn_606317?34X?U
N?K
=?:
8?5
inputs/0'??????????????????x?

 
p

 
? ".?+'??????????????????v??
+__inference_convLSTM_1_layer_call_fn_606326?34X?U
N?K
=?:
8?5
inputs/0'??????????????????x?

 
p 

 
? ".?+'??????????????????v??
+__inference_convLSTM_1_layer_call_fn_606737u34H?E
>?;
-?*
inputs?????????x?

 
p

 
? "%?"?????????v??
+__inference_convLSTM_1_layer_call_fn_606746u34H?E
>?;
-?*
inputs?????????x?

 
p 

 
? "%?"?????????v??
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607178?78W?T
M?J
<?9
7?4
inputs/0&??????????????????;O

 
p

 
? "-?*
#? 
0?????????9M 
? ?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607381?78W?T
M?J
<?9
7?4
inputs/0&??????????????????;O

 
p 

 
? "-?*
#? 
0?????????9M 
? ?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607602|78G?D
=?:
,?)
inputs?????????;O

 
p

 
? "-?*
#? 
0?????????9M 
? ?
F__inference_convLSTM_2_layer_call_and_return_conditional_losses_607805|78G?D
=?:
,?)
inputs?????????;O

 
p 

 
? "-?*
#? 
0?????????9M 
? ?
+__inference_convLSTM_2_layer_call_fn_60739078W?T
M?J
<?9
7?4
inputs/0&??????????????????;O

 
p

 
? " ??????????9M ?
+__inference_convLSTM_2_layer_call_fn_60739978W?T
M?J
<?9
7?4
inputs/0&??????????????????;O

 
p 

 
? " ??????????9M ?
+__inference_convLSTM_2_layer_call_fn_607814o78G?D
=?:
,?)
inputs?????????;O

 
p

 
? " ??????????9M ?
+__inference_convLSTM_2_layer_call_fn_607823o78G?D
=?:
,?)
inputs?????????;O

 
p 

 
? " ??????????9M ?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_608201?78???
???
(?%
inputs?????????;O
[?X
*?'
states/0?????????9M 
*?'
states/1?????????9M 
p
? "???
??~
%?"
0/0?????????9M 
U?R
'?$
0/1/0?????????9M 
'?$
0/1/1?????????9M 
? ?
O__inference_conv_lst_m2d_cell_1_layer_call_and_return_conditional_losses_608268?78???
???
(?%
inputs?????????;O
[?X
*?'
states/0?????????9M 
*?'
states/1?????????9M 
p 
? "???
??~
%?"
0/0?????????9M 
U?R
'?$
0/1/0?????????9M 
'?$
0/1/1?????????9M 
? ?
4__inference_conv_lst_m2d_cell_1_layer_call_fn_608283?78???
???
(?%
inputs?????????;O
[?X
*?'
states/0?????????9M 
*?'
states/1?????????9M 
p
? "{?x
#? 
0?????????9M 
Q?N
%?"
1/0?????????9M 
%?"
1/1?????????9M ?
4__inference_conv_lst_m2d_cell_1_layer_call_fn_608298?78???
???
(?%
inputs?????????;O
[?X
*?'
states/0?????????9M 
*?'
states/1?????????9M 
p 
? "{?x
#? 
0?????????9M 
Q?N
%?"
1/0?????????9M 
%?"
1/1?????????9M ?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_607911?34???
???
)?&
inputs?????????x?
]?Z
+?(
states/0?????????v?
+?(
states/1?????????v?
p
? "???
???
&?#
0/0?????????v?
W?T
(?%
0/1/0?????????v?
(?%
0/1/1?????????v?
? ?
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_607978?34???
???
)?&
inputs?????????x?
]?Z
+?(
states/0?????????v?
+?(
states/1?????????v?
p 
? "???
???
&?#
0/0?????????v?
W?T
(?%
0/1/0?????????v?
(?%
0/1/1?????????v?
? ?
2__inference_conv_lst_m2d_cell_layer_call_fn_607993?34???
???
)?&
inputs?????????x?
]?Z
+?(
states/0?????????v?
+?(
states/1?????????v?
p
? "~?{
$?!
0?????????v?
S?P
&?#
1/0?????????v?
&?#
1/1?????????v??
2__inference_conv_lst_m2d_cell_layer_call_fn_608008?34???
???
)?&
inputs?????????x?
]?Z
+?(
states/0?????????v?
+?(
states/1?????????v?
p 
? "~?{
$?!
0?????????v?
S?P
&?#
1/0?????????v?
&?#
1/1?????????v??
A__inference_dense_layer_call_and_return_conditional_losses_607834\()/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? y
&__inference_dense_layer_call_fn_607843O()/?,
%?"
 ?
inputs????????? 
? "???????????
H__inference_functional_1_layer_call_and_return_conditional_losses_604789|34??5678()E?B
;?8
.?+
input_1?????????x?
p

 
? "%?"
?
0?????????
? ?
H__inference_functional_1_layer_call_and_return_conditional_losses_604825|34??5678()E?B
;?8
.?+
input_1?????????x?
p 

 
? "%?"
?
0?????????
? ?
H__inference_functional_1_layer_call_and_return_conditional_losses_605421{34??5678()D?A
:?7
-?*
inputs?????????x?
p

 
? "%?"
?
0?????????
? ?
H__inference_functional_1_layer_call_and_return_conditional_losses_605855{34??5678()D?A
:?7
-?*
inputs?????????x?
p 

 
? "%?"
?
0?????????
? ?
-__inference_functional_1_layer_call_fn_604887o34??5678()E?B
;?8
.?+
input_1?????????x?
p

 
? "???????????
-__inference_functional_1_layer_call_fn_604948o34??5678()E?B
;?8
.?+
input_1?????????x?
p 

 
? "???????????
-__inference_functional_1_layer_call_fn_605880n34??5678()D?A
:?7
-?*
inputs?????????x?
p

 
? "???????????
-__inference_functional_1_layer_call_fn_605905n34??5678()D?A
:?7
-?*
inputs?????????x?
p 

 
? "???????????
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_603784?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
5__inference_global_max_pooling2d_layer_call_fn_603790wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!???????????????????8
__inference_loss_fn_0_608013?

? 
? "? 8
__inference_loss_fn_1_608303?

? 
? "? ?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_602809?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_602815?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
$__inference_signature_wrapper_604985?34??5678()H?E
? 
>?;
9
input_1.?+
input_1?????????x?"-?*
(
dense?
dense??????????
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606851???56L?I
B??
5?2
inputs&??????????????????;O
p

 
? ":?7
0?-
0&??????????????????;O
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606880???56L?I
B??
5?2
inputs&??????????????????;O
p 

 
? ":?7
0?-
0&??????????????????;O
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606928???56C?@
9?6
,?)
inputs?????????;O
p

 
? "1?.
'?$
0?????????;O
? ?
N__inference_time_distributed_1_layer_call_and_return_conditional_losses_606948???56C?@
9?6
,?)
inputs?????????;O
p 

 
? "1?.
'?$
0?????????;O
? ?
3__inference_time_distributed_1_layer_call_fn_606893???56L?I
B??
5?2
inputs&??????????????????;O
p

 
? "-?*&??????????????????;O?
3__inference_time_distributed_1_layer_call_fn_606906???56L?I
B??
5?2
inputs&??????????????????;O
p 

 
? "-?*&??????????????????;O?
3__inference_time_distributed_1_layer_call_fn_606961s??56C?@
9?6
,?)
inputs?????????;O
p

 
? "$?!?????????;O?
3__inference_time_distributed_1_layer_call_fn_606974s??56C?@
9?6
,?)
inputs?????????;O
p 

 
? "$?!?????????;O?
L__inference_time_distributed_layer_call_and_return_conditional_losses_606755yD?A
:?7
-?*
inputs?????????v?
p

 
? "1?.
'?$
0?????????;O
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_606764yD?A
:?7
-?*
inputs?????????v?
p 

 
? "1?.
'?$
0?????????;O
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_606792?M?J
C?@
6?3
inputs'??????????????????v?
p

 
? ":?7
0?-
0&??????????????????;O
? ?
L__inference_time_distributed_layer_call_and_return_conditional_losses_606810?M?J
C?@
6?3
inputs'??????????????????v?
p 

 
? ":?7
0?-
0&??????????????????;O
? ?
1__inference_time_distributed_layer_call_fn_606769lD?A
:?7
-?*
inputs?????????v?
p

 
? "$?!?????????;O?
1__inference_time_distributed_layer_call_fn_606774lD?A
:?7
-?*
inputs?????????v?
p 

 
? "$?!?????????;O?
1__inference_time_distributed_layer_call_fn_606815~M?J
C?@
6?3
inputs'??????????????????v?
p

 
? "-?*&??????????????????;O?
1__inference_time_distributed_layer_call_fn_606820~M?J
C?@
6?3
inputs'??????????????????v?
p 

 
? "-?*&??????????????????;O