©
¿£
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
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8ûÄ

batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_1/moving_mean

+batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_mean*
_output_shapes
:*
dtype0

batchnorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_1/moving_variance

/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_variance*
_output_shapes
:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
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

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

convLSTM_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameconvLSTM_1/recurrent_kernel

/convLSTM_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpconvLSTM_1/recurrent_kernel*&
_output_shapes
:@*
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
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
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

Adam/convLSTM_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/convLSTM_1/kernel/m

,Adam/convLSTM_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/convLSTM_1/kernel/m*&
_output_shapes
:@*
dtype0
¨
"Adam/convLSTM_1/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/convLSTM_1/recurrent_kernel/m
¡
6Adam/convLSTM_1/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp"Adam/convLSTM_1/recurrent_kernel/m*&
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:*
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

Adam/convLSTM_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/convLSTM_1/kernel/v

,Adam/convLSTM_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/convLSTM_1/kernel/v*&
_output_shapes
:@*
dtype0
¨
"Adam/convLSTM_1/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/convLSTM_1/recurrent_kernel/v
¡
6Adam/convLSTM_1/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp"Adam/convLSTM_1/recurrent_kernel/v*&
_output_shapes
:@*
dtype0

ConstConst*
_output_shapes
:*
dtype0*U
valueLBJ"@  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?

Const_1Const*
_output_shapes
:*
dtype0*U
valueLBJ"@                                                                

NoOpNoOp
¬+
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*å*
valueÛ*BØ* BÑ*

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
 
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api

axis
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api

#iter

$beta_1

%beta_2
	&decay
'learning_ratem_m`(ma)mbvcvd(ve)vf

(0
)1
2
3
 
*
(0
)1
2
3
4
5
­
trainable_variables
regularization_losses
*layer_regularization_losses

+layers
,layer_metrics
-metrics
.non_trainable_variables
		variables
 
t

(kernel
)recurrent_kernel
/regularization_losses
0trainable_variables
1	variables
2	keras_api
 

(0
)1
 

(0
)1
¹

3states
trainable_variables
regularization_losses
4layer_regularization_losses

5layers
6layer_metrics
7metrics
8non_trainable_variables
	variables
 
hf
VARIABLE_VALUEbatchnorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
­
9layer_regularization_losses
regularization_losses
trainable_variables

:layers
;layer_metrics
<metrics
=non_trainable_variables
	variables
 
 
 
­
>layer_regularization_losses
regularization_losses
trainable_variables

?layers
@layer_metrics
Ametrics
Bnon_trainable_variables
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
Clayer_regularization_losses
regularization_losses
 trainable_variables

Dlayers
Elayer_metrics
Fmetrics
Gnon_trainable_variables
!	variables
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
WU
VARIABLE_VALUEconvLSTM_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconvLSTM_1/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

H0
I1
J2

0
1
 

(0
)1

(0
)1
­
Klayer_regularization_losses
/regularization_losses
0trainable_variables

Llayers
Mlayer_metrics
Nmetrics
Onon_trainable_variables
1	variables
 
 

0
 
 
 
 
 
 
 

0
1
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
4
	Ptotal
	Qcount
R	variables
S	keras_api
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
p
Ytrue_positives
Ztrue_negatives
[false_positives
\false_negatives
]	variables
^	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
[2
\3

]	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/convLSTM_1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/convLSTM_1/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/convLSTM_1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/convLSTM_1/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx *
dtype0*)
shape :ÿÿÿÿÿÿÿÿÿx 
Ó
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1convLSTM_1/kernelconvLSTM_1/recurrent_kernelConstConst_1batchnorm_1/moving_meanbatchnorm_1/moving_variancedense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_577785
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ê

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+batchnorm_1/moving_mean/Read/ReadVariableOp/batchnorm_1/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp%convLSTM_1/kernel/Read/ReadVariableOp/convLSTM_1/recurrent_kernel/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp,Adam/convLSTM_1/kernel/m/Read/ReadVariableOp6Adam/convLSTM_1/recurrent_kernel/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp,Adam/convLSTM_1/kernel/v/Read/ReadVariableOp6Adam/convLSTM_1/recurrent_kernel/v/Read/ReadVariableOpConst_2*(
Tin!
2	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_579530
§
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatchnorm_1/moving_meanbatchnorm_1/moving_variancedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconvLSTM_1/kernelconvLSTM_1/recurrent_kerneltotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativesAdam/dense/kernel/mAdam/dense/bias/mAdam/convLSTM_1/kernel/m"Adam/convLSTM_1/recurrent_kernel/mAdam/dense/kernel/vAdam/dense/bias/vAdam/convLSTM_1/kernel/v"Adam/convLSTM_1/recurrent_kernel/v*'
Tin 
2*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_579621¯Í


,__inference_batchnorm_1_layer_call_fn_579221

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5770542
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
©
A__inference_dense_layer_call_and_return_conditional_losses_577617

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ;
ð
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_579389

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2P
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1 
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution¤
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¤
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¤
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3§
convolution_4Conv2Dstates_0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_4§
convolution_5Conv2Dstates_0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5§
convolution_6Conv2Dstates_0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6§
convolution_7Conv2Dstates_0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
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
:ÿÿÿÿÿÿÿÿÿv2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿx :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/1
ç6
£
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_576991

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall¢whileu

zeros_like	ZerosLikeinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
shrink_axis_mask2
strided_slice_1
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_5765742
StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_576926*
condR
while_cond_576925*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeú
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm·
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const
IdentityIdentitystrided_slice_2:output:0^StatefulPartitionedCall^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx ::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
Æ

Ñ
convLSTM_1_while_cond_5778692
.convlstm_1_while_convlstm_1_while_loop_counter8
4convlstm_1_while_convlstm_1_while_maximum_iterations 
convlstm_1_while_placeholder"
convlstm_1_while_placeholder_1"
convlstm_1_while_placeholder_2"
convlstm_1_while_placeholder_32
.convlstm_1_while_less_convlstm_1_strided_sliceJ
Fconvlstm_1_while_convlstm_1_while_cond_577869___redundant_placeholder0J
Fconvlstm_1_while_convlstm_1_while_cond_577869___redundant_placeholder1J
Fconvlstm_1_while_convlstm_1_while_cond_577869___redundant_placeholder2
convlstm_1_while_identity
¥
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:


while_cond_576817
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_576817___redundant_placeholder04
0while_while_cond_576817___redundant_placeholder14
0while_while_cond_576817___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
¨
©
A__inference_dense_layer_call_and_return_conditional_losses_579245

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

+__inference_convLSTM_1_layer_call_fn_579105
inputs_0
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5768832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx ::22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 
"
_user_specified_name
inputs/0

Ã
2__inference_conv_lst_m2d_cell_layer_call_fn_579419

inputs
states_0
states_1
unknown
	unknown_0
identity

identity_1

identity_2¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_5765742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿx :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/1
èe
µ
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578475

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity¢whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1¶
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¶
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¶
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¶
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_4³
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5³
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6³
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7³
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_578359*
condR
while_cond_578358*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const}
IdentityIdentitystrided_slice_2:output:0^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿx ::2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
§"

while_body_576926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_576950_0
while_576952_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_576950
while_576952¢while/StatefulPartitionedCallË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem±
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_576950_0while_576952_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_5765742
while/StatefulPartitionedCallê
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
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2­
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3³
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4³
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5"
while_576950while_576950_0"
while_576952while_576952_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::2>
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
>
¦
__inference__traced_save_579530
file_prefix6
2savev2_batchnorm_1_moving_mean_read_readvariableop:
6savev2_batchnorm_1_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop0
,savev2_convlstm_1_kernel_read_readvariableop:
6savev2_convlstm_1_recurrent_kernel_read_readvariableop$
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
=savev2_adam_convlstm_1_recurrent_kernel_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop7
3savev2_adam_convlstm_1_kernel_v_read_readvariableopA
=savev2_adam_convlstm_1_recurrent_kernel_v_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_99f04010efa84849ab40791c571d8952/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename°
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Â
value¸BµB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÀ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices£
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_batchnorm_1_moving_mean_read_readvariableop6savev2_batchnorm_1_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop,savev2_convlstm_1_kernel_read_readvariableop6savev2_convlstm_1_recurrent_kernel_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop3savev2_adam_convlstm_1_kernel_m_read_readvariableop=savev2_adam_convlstm_1_recurrent_kernel_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop3savev2_adam_convlstm_1_kernel_v_read_readvariableop=savev2_adam_convlstm_1_recurrent_kernel_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*ï
_input_shapesÝ
Ú: ::::: : : : : :@:@: : : : :È:È:È:È:::@:@:::@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :,
(
&
_output_shapes
:@:,(
&
_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@:,(
&
_output_shapes
:@:

_output_shapes
: 

Ò
H__inference_functional_1_layer_call_and_return_conditional_losses_578231

inputs,
(convlstm_1_split_readvariableop_resource.
*convlstm_1_split_1_readvariableop_resource
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢convLSTM_1/while
convLSTM_1/zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
convLSTM_1/zeros_like
 convLSTM_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/Sum/reduction_indices¨
convLSTM_1/SumSumconvLSTM_1/zeros_like:y:0)convLSTM_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 2
convLSTM_1/Sum
convLSTM_1/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
convLSTM_1/zerosÒ
convLSTM_1/convolutionConv2DconvLSTM_1/Sum:output:0convLSTM_1/zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution
convLSTM_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose/perm¤
convLSTM_1/transpose	Transposeinputs"convLSTM_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
convLSTM_1/transposel
convLSTM_1/ShapeShapeconvLSTM_1/transpose:y:0*
T0*
_output_shapes
:2
convLSTM_1/Shape
convLSTM_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
convLSTM_1/strided_slice/stack
 convLSTM_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_1
 convLSTM_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_2¤
convLSTM_1/strided_sliceStridedSliceconvLSTM_1/Shape:output:0'convLSTM_1/strided_slice/stack:output:0)convLSTM_1/strided_slice/stack_1:output:0)convLSTM_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
convLSTM_1/strided_slice
&convLSTM_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&convLSTM_1/TensorArrayV2/element_shapeÜ
convLSTM_1/TensorArrayV2TensorListReserve/convLSTM_1/TensorArrayV2/element_shape:output:0!convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_1/TensorArrayV2Ý
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          2B
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2convLSTM_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconvLSTM_1/transpose:y:0IconvLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2convLSTM_1/TensorArrayUnstack/TensorListFromTensor
 convLSTM_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 convLSTM_1/strided_slice_1/stack
"convLSTM_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_1
"convLSTM_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_2Ç
convLSTM_1/strided_slice_1StridedSliceconvLSTM_1/transpose:y:0)convLSTM_1/strided_slice_1/stack:output:0+convLSTM_1/strided_slice_1/stack_1:output:0+convLSTM_1/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
convLSTM_1/split/split_dim³
convLSTM_1/split/ReadVariableOpReadVariableOp(convlstm_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype02!
convLSTM_1/split/ReadVariableOpó
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
convLSTM_1/split_1/split_dim¹
!convLSTM_1/split_1/ReadVariableOpReadVariableOp*convlstm_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!convLSTM_1/split_1/ReadVariableOpû
convLSTM_1/split_1Split%convLSTM_1/split_1/split_dim:output:0)convLSTM_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/split_1â
convLSTM_1/convolution_1Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_1â
convLSTM_1/convolution_2Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_2â
convLSTM_1/convolution_3Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_3â
convLSTM_1/convolution_4Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_4ß
convLSTM_1/convolution_5Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_5ß
convLSTM_1/convolution_6Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_6ß
convLSTM_1/convolution_7Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_7ß
convLSTM_1/convolution_8Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_8ª
convLSTM_1/addAddV2!convLSTM_1/convolution_1:output:0!convLSTM_1/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/addm
convLSTM_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/Const_2m
convLSTM_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_3
convLSTM_1/MulMulconvLSTM_1/add:z:0convLSTM_1/Const_2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Mul
convLSTM_1/Add_1AddconvLSTM_1/Mul:z:0convLSTM_1/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Add_1
"convLSTM_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"convLSTM_1/clip_by_value/Minimum/yÍ
 convLSTM_1/clip_by_value/MinimumMinimumconvLSTM_1/Add_1:z:0+convLSTM_1/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2"
 convLSTM_1/clip_by_value/Minimum}
convLSTM_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value/yÅ
convLSTM_1/clip_by_valueMaximum$convLSTM_1/clip_by_value/Minimum:z:0#convLSTM_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/clip_by_value®
convLSTM_1/add_2AddV2!convLSTM_1/convolution_2:output:0!convLSTM_1/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_2m
convLSTM_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/Const_4m
convLSTM_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_5
convLSTM_1/Mul_1MulconvLSTM_1/add_2:z:0convLSTM_1/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Mul_1
convLSTM_1/Add_3AddconvLSTM_1/Mul_1:z:0convLSTM_1/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Add_3
$convLSTM_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$convLSTM_1/clip_by_value_1/Minimum/yÓ
"convLSTM_1/clip_by_value_1/MinimumMinimumconvLSTM_1/Add_3:z:0-convLSTM_1/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2$
"convLSTM_1/clip_by_value_1/Minimum
convLSTM_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_1/yÍ
convLSTM_1/clip_by_value_1Maximum&convLSTM_1/clip_by_value_1/Minimum:z:0%convLSTM_1/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/clip_by_value_1§
convLSTM_1/mul_2MulconvLSTM_1/clip_by_value_1:z:0convLSTM_1/convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/mul_2®
convLSTM_1/add_4AddV2!convLSTM_1/convolution_3:output:0!convLSTM_1/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_4{
convLSTM_1/TanhTanhconvLSTM_1/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Tanh
convLSTM_1/mul_3MulconvLSTM_1/clip_by_value:z:0convLSTM_1/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/mul_3
convLSTM_1/add_5AddV2convLSTM_1/mul_2:z:0convLSTM_1/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_5®
convLSTM_1/add_6AddV2!convLSTM_1/convolution_4:output:0!convLSTM_1/convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_6m
convLSTM_1/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/Const_6m
convLSTM_1/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_7
convLSTM_1/Mul_4MulconvLSTM_1/add_6:z:0convLSTM_1/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Mul_4
convLSTM_1/Add_7AddconvLSTM_1/Mul_4:z:0convLSTM_1/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Add_7
$convLSTM_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$convLSTM_1/clip_by_value_2/Minimum/yÓ
"convLSTM_1/clip_by_value_2/MinimumMinimumconvLSTM_1/Add_7:z:0-convLSTM_1/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2$
"convLSTM_1/clip_by_value_2/Minimum
convLSTM_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_2/yÍ
convLSTM_1/clip_by_value_2Maximum&convLSTM_1/clip_by_value_2/Minimum:z:0%convLSTM_1/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/clip_by_value_2
convLSTM_1/Tanh_1TanhconvLSTM_1/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Tanh_1
convLSTM_1/mul_5MulconvLSTM_1/clip_by_value_2:z:0convLSTM_1/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/mul_5­
(convLSTM_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2*
(convLSTM_1/TensorArrayV2_1/element_shapeâ
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
convLSTM_1/time
#convLSTM_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#convLSTM_1/while/maximum_iterations
convLSTM_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_1/while/loop_counterä
convLSTM_1/whileWhile&convLSTM_1/while/loop_counter:output:0,convLSTM_1/while/maximum_iterations:output:0convLSTM_1/time:output:0#convLSTM_1/TensorArrayV2_1:handle:0convLSTM_1/convolution:output:0convLSTM_1/convolution:output:0!convLSTM_1/strided_slice:output:0BconvLSTM_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(convlstm_1_split_readvariableop_resource*convlstm_1_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*(
body R
convLSTM_1_while_body_578094*(
cond R
convLSTM_1_while_cond_578093*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
convLSTM_1/whileÓ
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2=
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape
-convLSTM_1/TensorArrayV2Stack/TensorListStackTensorListStackconvLSTM_1/while:output:3DconvLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02/
-convLSTM_1/TensorArrayV2Stack/TensorListStack
 convLSTM_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 convLSTM_1/strided_slice_2/stack
"convLSTM_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"convLSTM_1/strided_slice_2/stack_1
"convLSTM_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_2/stack_2å
convLSTM_1/strided_slice_2StridedSlice6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0)convLSTM_1/strided_slice_2/stack:output:0+convLSTM_1/strided_slice_2/stack_1:output:0+convLSTM_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
convLSTM_1/strided_slice_2
convLSTM_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose_1/permÚ
convLSTM_1/transpose_1	Transpose6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0$convLSTM_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/transpose_1Ë
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1
batchnorm_1/FusedBatchNormV3FusedBatchNormV3#convLSTM_1/strided_slice_2:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
is_training( 2
batchnorm_1/FusedBatchNormV3©
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesÄ
global_max_pooling2d/MaxMax batchnorm_1/FusedBatchNormV3:y:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_max_pooling2d/Max
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp 
dense/MatMulMatMul!global_max_pooling2d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Sigmoid
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constx
IdentityIdentitydense/Sigmoid:y:0^convLSTM_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::2$
convLSTM_1/whileconvLSTM_1/while:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
ã;
î
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_576507

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2P
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1 
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution¤
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¤
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¤
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¥
convolution_4Conv2Dstatessplit_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_4¥
convolution_5Conv2Dstatessplit_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5¥
convolution_6Conv2Dstatessplit_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6¥
convolution_7Conv2Dstatessplit_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
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
:ÿÿÿÿÿÿÿÿÿv2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿx :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_namestates:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_namestates

ÿ
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579192

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ç
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
{
&__inference_dense_layer_call_fn_579254

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5776172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ;
ð
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_579322

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2P
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1 
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution¤
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¤
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¤
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3§
convolution_4Conv2Dstates_0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_4§
convolution_5Conv2Dstates_0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5§
convolution_6Conv2Dstates_0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6§
convolution_7Conv2Dstates_0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
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
:ÿÿÿÿÿÿÿÿÿv2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿx :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/1


while_cond_576925
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_576925___redundant_placeholder04
0while_while_cond_576925___redundant_placeholder14
0while_while_cond_576925___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
¨

+__inference_convLSTM_1_layer_call_fn_579114
inputs_0
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5769912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx ::22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 
"
_user_specified_name
inputs/0
¦f
·
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578895
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
identity¢whilew

zeros_like	ZerosLikeinputs_0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1¶
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¶
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¶
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¶
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_4³
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5³
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6³
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7³
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_578779*
condR
while_cond_578778*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeú
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm·
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const}
IdentityIdentitystrided_slice_2:output:0^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx ::2
whilewhile:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 
"
_user_specified_name
inputs/0
¬X
Ì
while_body_577195
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
%while_split_1_readvariableop_resourceË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
while/split/split_dim¦
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOpß
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
while/split_1/split_dim¬
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOpç
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1Ü
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolutionà
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_1à
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_2à
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_3Ä
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_4Ä
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_5Ä
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_6Ä
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_7
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
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
:ÿÿÿÿÿÿÿÿÿv2
	while/Mul
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_1
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/clip_by_value/Minimum/y¹
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y±
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_1
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_3
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_1/Minimum/y¿
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y¹
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_2
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

while/Tanh
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_5
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_4
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_7
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_2/Minimum/y¿
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y¹
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Tanh_1
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_5Ó
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 


while_cond_578358
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_578358___redundant_placeholder04
0while_while_cond_578358___redundant_placeholder14
0while_while_cond_578358___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
Ì

,__inference_batchnorm_1_layer_call_fn_579174

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5775692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿv::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs


while_cond_578778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_578778___redundant_placeholder04
0while_while_cond_578778___redundant_placeholder14
0while_while_cond_578778___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
Å
Ý
-__inference_functional_1_layer_call_fn_577707
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5776882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
!
_user_specified_name	input_1
Æ

Ñ
convLSTM_1_while_cond_5780932
.convlstm_1_while_convlstm_1_while_loop_counter8
4convlstm_1_while_convlstm_1_while_maximum_iterations 
convlstm_1_while_placeholder"
convlstm_1_while_placeholder_1"
convlstm_1_while_placeholder_2"
convlstm_1_while_placeholder_32
.convlstm_1_while_less_convlstm_1_strided_sliceJ
Fconvlstm_1_while_convlstm_1_while_cond_578093___redundant_placeholder0J
Fconvlstm_1_while_convlstm_1_while_cond_578093___redundant_placeholder1J
Fconvlstm_1_while_convlstm_1_while_cond_578093___redundant_placeholder2
convlstm_1_while_identity
¥
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
¬X
Ì
while_body_578560
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
%while_split_1_readvariableop_resourceË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
while/split/split_dim¦
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOpß
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
while/split_1/split_dim¬
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOpç
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1Ü
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolutionà
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_1à
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_2à
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_3Ä
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_4Ä
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_5Ä
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_6Ä
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_7
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
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
:ÿÿÿÿÿÿÿÿÿv2
	while/Mul
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_1
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/clip_by_value/Minimum/y¹
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y±
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_1
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_3
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_1/Minimum/y¿
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y¹
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_2
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

while/Tanh
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_5
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_4
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_7
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_2/Minimum/y¿
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y¹
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Tanh_1
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_5Ó
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
­

H__inference_functional_1_layer_call_and_return_conditional_losses_577688

inputs
convlstm_1_577666
convlstm_1_577668
batchnorm_1_577671
batchnorm_1_577673
batchnorm_1_577675
batchnorm_1_577677
dense_577681
dense_577683
identity¢#batchnorm_1/StatefulPartitionedCall¢"convLSTM_1/StatefulPartitionedCall¢dense/StatefulPartitionedCallª
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinputsconvlstm_1_577666convlstm_1_577668*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5773112$
"convLSTM_1/StatefulPartitionedCallú
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0batchnorm_1_577671batchnorm_1_577673batchnorm_1_577675batchnorm_1_577677*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5775532%
#batchnorm_1/StatefulPartitionedCall¡
$global_max_pooling2d/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_5771012&
$global_max_pooling2d/PartitionedCall¯
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_577681dense_577683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5776172
dense/StatefulPartitionedCall
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constå
IdentityIdentity&dense/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall#^convLSTM_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
Å
ÿ
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_577553

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¶
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿv::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs

ÿ
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_577054

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ç
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


+__inference_convLSTM_1_layer_call_fn_578694

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5775122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿx ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
	
Û
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_577569

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¨
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿv:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
æn
õ
convLSTM_1_while_body_5780942
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
0convlstm_1_while_split_1_readvariableop_resourceá
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          2D
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0convlstm_1_while_placeholderKconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
element_dtype026
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemr
convLSTM_1/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/Const
 convLSTM_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/while/split/split_dimÇ
%convLSTM_1/while/split/ReadVariableOpReadVariableOp0convlstm_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02'
%convLSTM_1/while/split/ReadVariableOp
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
convLSTM_1/while/Const_1
"convLSTM_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"convLSTM_1/while/split_1/split_dimÍ
'convLSTM_1/while/split_1/ReadVariableOpReadVariableOp2convlstm_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02)
'convLSTM_1/while/split_1/ReadVariableOp
convLSTM_1/while/split_1Split+convLSTM_1/while/split_1/split_dim:output:0/convLSTM_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/while/split_1
convLSTM_1/while/convolutionConv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/while/convolution
convLSTM_1/while/convolution_1Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_1
convLSTM_1/while/convolution_2Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_2
convLSTM_1/while/convolution_3Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_3ð
convLSTM_1/while/convolution_4Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_4ð
convLSTM_1/while/convolution_5Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_5ð
convLSTM_1/while/convolution_6Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_6ð
convLSTM_1/while/convolution_7Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_7À
convLSTM_1/while/addAddV2%convLSTM_1/while/convolution:output:0'convLSTM_1/while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/addy
convLSTM_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/while/Const_2y
convLSTM_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_3«
convLSTM_1/while/MulMulconvLSTM_1/while/add:z:0!convLSTM_1/while/Const_2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Mul¯
convLSTM_1/while/Add_1AddconvLSTM_1/while/Mul:z:0!convLSTM_1/while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Add_1
(convLSTM_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(convLSTM_1/while/clip_by_value/Minimum/yå
&convLSTM_1/while/clip_by_value/MinimumMinimumconvLSTM_1/while/Add_1:z:01convLSTM_1/while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2(
&convLSTM_1/while/clip_by_value/Minimum
 convLSTM_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 convLSTM_1/while/clip_by_value/yÝ
convLSTM_1/while/clip_by_valueMaximum*convLSTM_1/while/clip_by_value/Minimum:z:0)convLSTM_1/while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2 
convLSTM_1/while/clip_by_valueÆ
convLSTM_1/while/add_2AddV2'convLSTM_1/while/convolution_1:output:0'convLSTM_1/while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_2y
convLSTM_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/while/Const_4y
convLSTM_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_5±
convLSTM_1/while/Mul_1MulconvLSTM_1/while/add_2:z:0!convLSTM_1/while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Mul_1±
convLSTM_1/while/Add_3AddconvLSTM_1/while/Mul_1:z:0!convLSTM_1/while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Add_3
*convLSTM_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*convLSTM_1/while/clip_by_value_1/Minimum/yë
(convLSTM_1/while/clip_by_value_1/MinimumMinimumconvLSTM_1/while/Add_3:z:03convLSTM_1/while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2*
(convLSTM_1/while/clip_by_value_1/Minimum
"convLSTM_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_1/yå
 convLSTM_1/while/clip_by_value_1Maximum,convLSTM_1/while/clip_by_value_1/Minimum:z:0+convLSTM_1/while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2"
 convLSTM_1/while/clip_by_value_1¸
convLSTM_1/while/mul_2Mul$convLSTM_1/while/clip_by_value_1:z:0convlstm_1_while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/mul_2Æ
convLSTM_1/while/add_4AddV2'convLSTM_1/while/convolution_2:output:0'convLSTM_1/while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_4
convLSTM_1/while/TanhTanhconvLSTM_1/while/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Tanh±
convLSTM_1/while/mul_3Mul"convLSTM_1/while/clip_by_value:z:0convLSTM_1/while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/mul_3¬
convLSTM_1/while/add_5AddV2convLSTM_1/while/mul_2:z:0convLSTM_1/while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_5Æ
convLSTM_1/while/add_6AddV2'convLSTM_1/while/convolution_3:output:0'convLSTM_1/while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_6y
convLSTM_1/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/while/Const_6y
convLSTM_1/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_7±
convLSTM_1/while/Mul_4MulconvLSTM_1/while/add_6:z:0!convLSTM_1/while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Mul_4±
convLSTM_1/while/Add_7AddconvLSTM_1/while/Mul_4:z:0!convLSTM_1/while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Add_7
*convLSTM_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*convLSTM_1/while/clip_by_value_2/Minimum/yë
(convLSTM_1/while/clip_by_value_2/MinimumMinimumconvLSTM_1/while/Add_7:z:03convLSTM_1/while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2*
(convLSTM_1/while/clip_by_value_2/Minimum
"convLSTM_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_2/yå
 convLSTM_1/while/clip_by_value_2Maximum,convLSTM_1/while/clip_by_value_2/Minimum:z:0+convLSTM_1/while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2"
 convLSTM_1/while/clip_by_value_2
convLSTM_1/while/Tanh_1TanhconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Tanh_1µ
convLSTM_1/while/mul_5Mul$convLSTM_1/while/clip_by_value_2:z:0convLSTM_1/while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/mul_5
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
convLSTM_1/while/add_8/y
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
convLSTM_1/while/add_9/y­
convLSTM_1/while/add_9AddV2.convlstm_1_while_convlstm_1_while_loop_counter!convLSTM_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2
convLSTM_1/while/add_9
convLSTM_1/while/IdentityIdentityconvLSTM_1/while/add_9:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity
convLSTM_1/while/Identity_1Identity4convlstm_1_while_convlstm_1_while_maximum_iterations*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_1
convLSTM_1/while/Identity_2IdentityconvLSTM_1/while/add_8:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_2®
convLSTM_1/while/Identity_3IdentityEconvLSTM_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_3
convLSTM_1/while/Identity_4IdentityconvLSTM_1/while/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Identity_4
convLSTM_1/while/Identity_5IdentityconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Identity_5"X
)convlstm_1_while_convlstm_1_strided_slice+convlstm_1_while_convlstm_1_strided_slice_0"?
convlstm_1_while_identity"convLSTM_1/while/Identity:output:0"C
convlstm_1_while_identity_1$convLSTM_1/while/Identity_1:output:0"C
convlstm_1_while_identity_2$convLSTM_1/while/Identity_2:output:0"C
convlstm_1_while_identity_3$convLSTM_1/while/Identity_3:output:0"C
convlstm_1_while_identity_4$convLSTM_1/while/Identity_4:output:0"C
convlstm_1_while_identity_5$convLSTM_1/while/Identity_5:output:0"f
0convlstm_1_while_split_1_readvariableop_resource2convlstm_1_while_split_1_readvariableop_resource_0"b
.convlstm_1_while_split_readvariableop_resource0convlstm_1_while_split_readvariableop_resource_0"Ô
gconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensoriconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
Ä
Ü
-__inference_functional_1_layer_call_fn_578273

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5777342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
Â
Ü
-__inference_functional_1_layer_call_fn_578252

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5776882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs


,__inference_batchnorm_1_layer_call_fn_579234

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5770832
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ìr
þ
"__inference__traced_restore_579621
file_prefix,
(assignvariableop_batchnorm_1_moving_mean2
.assignvariableop_1_batchnorm_1_moving_variance#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate(
$assignvariableop_9_convlstm_1_kernel3
/assignvariableop_10_convlstm_1_recurrent_kernel
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1&
"assignvariableop_15_true_positives&
"assignvariableop_16_true_negatives'
#assignvariableop_17_false_positives'
#assignvariableop_18_false_negatives+
'assignvariableop_19_adam_dense_kernel_m)
%assignvariableop_20_adam_dense_bias_m0
,assignvariableop_21_adam_convlstm_1_kernel_m:
6assignvariableop_22_adam_convlstm_1_recurrent_kernel_m+
'assignvariableop_23_adam_dense_kernel_v)
%assignvariableop_24_adam_dense_bias_v0
,assignvariableop_25_adam_convlstm_1_kernel_v:
6assignvariableop_26_adam_convlstm_1_recurrent_kernel_v
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¶
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Â
value¸BµB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¸
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity§
AssignVariableOpAssignVariableOp(assignvariableop_batchnorm_1_moving_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1³
AssignVariableOp_1AssignVariableOp.assignvariableop_1_batchnorm_1_moving_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¡
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_convlstm_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_convlstm_1_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14£
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ª
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ª
AssignVariableOp_16AssignVariableOp"assignvariableop_16_true_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¯
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20­
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21´
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_convlstm_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¾
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_convlstm_1_recurrent_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¯
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24­
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25´
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_convlstm_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_convlstm_1_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp°
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27£
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*
_input_shapesp
n: :::::::::::::::::::::::::::2$
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
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
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
®
l
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_577101

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
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Maxi
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æn
õ
convLSTM_1_while_body_5778702
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
0convlstm_1_while_split_1_readvariableop_resourceá
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          2D
BconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemiconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0convlstm_1_while_placeholderKconvLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
element_dtype026
4convLSTM_1/while/TensorArrayV2Read/TensorListGetItemr
convLSTM_1/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
convLSTM_1/while/Const
 convLSTM_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/while/split/split_dimÇ
%convLSTM_1/while/split/ReadVariableOpReadVariableOp0convlstm_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02'
%convLSTM_1/while/split/ReadVariableOp
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
convLSTM_1/while/Const_1
"convLSTM_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"convLSTM_1/while/split_1/split_dimÍ
'convLSTM_1/while/split_1/ReadVariableOpReadVariableOp2convlstm_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02)
'convLSTM_1/while/split_1/ReadVariableOp
convLSTM_1/while/split_1Split+convLSTM_1/while/split_1/split_dim:output:0/convLSTM_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/while/split_1
convLSTM_1/while/convolutionConv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/while/convolution
convLSTM_1/while/convolution_1Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_1
convLSTM_1/while/convolution_2Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_2
convLSTM_1/while/convolution_3Conv2D;convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0convLSTM_1/while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2 
convLSTM_1/while/convolution_3ð
convLSTM_1/while/convolution_4Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_4ð
convLSTM_1/while/convolution_5Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_5ð
convLSTM_1/while/convolution_6Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_6ð
convLSTM_1/while/convolution_7Conv2Dconvlstm_1_while_placeholder_2!convLSTM_1/while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2 
convLSTM_1/while/convolution_7À
convLSTM_1/while/addAddV2%convLSTM_1/while/convolution:output:0'convLSTM_1/while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/addy
convLSTM_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/while/Const_2y
convLSTM_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_3«
convLSTM_1/while/MulMulconvLSTM_1/while/add:z:0!convLSTM_1/while/Const_2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Mul¯
convLSTM_1/while/Add_1AddconvLSTM_1/while/Mul:z:0!convLSTM_1/while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Add_1
(convLSTM_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(convLSTM_1/while/clip_by_value/Minimum/yå
&convLSTM_1/while/clip_by_value/MinimumMinimumconvLSTM_1/while/Add_1:z:01convLSTM_1/while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2(
&convLSTM_1/while/clip_by_value/Minimum
 convLSTM_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 convLSTM_1/while/clip_by_value/yÝ
convLSTM_1/while/clip_by_valueMaximum*convLSTM_1/while/clip_by_value/Minimum:z:0)convLSTM_1/while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2 
convLSTM_1/while/clip_by_valueÆ
convLSTM_1/while/add_2AddV2'convLSTM_1/while/convolution_1:output:0'convLSTM_1/while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_2y
convLSTM_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/while/Const_4y
convLSTM_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_5±
convLSTM_1/while/Mul_1MulconvLSTM_1/while/add_2:z:0!convLSTM_1/while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Mul_1±
convLSTM_1/while/Add_3AddconvLSTM_1/while/Mul_1:z:0!convLSTM_1/while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Add_3
*convLSTM_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*convLSTM_1/while/clip_by_value_1/Minimum/yë
(convLSTM_1/while/clip_by_value_1/MinimumMinimumconvLSTM_1/while/Add_3:z:03convLSTM_1/while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2*
(convLSTM_1/while/clip_by_value_1/Minimum
"convLSTM_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_1/yå
 convLSTM_1/while/clip_by_value_1Maximum,convLSTM_1/while/clip_by_value_1/Minimum:z:0+convLSTM_1/while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2"
 convLSTM_1/while/clip_by_value_1¸
convLSTM_1/while/mul_2Mul$convLSTM_1/while/clip_by_value_1:z:0convlstm_1_while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/mul_2Æ
convLSTM_1/while/add_4AddV2'convLSTM_1/while/convolution_2:output:0'convLSTM_1/while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_4
convLSTM_1/while/TanhTanhconvLSTM_1/while/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Tanh±
convLSTM_1/while/mul_3Mul"convLSTM_1/while/clip_by_value:z:0convLSTM_1/while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/mul_3¬
convLSTM_1/while/add_5AddV2convLSTM_1/while/mul_2:z:0convLSTM_1/while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_5Æ
convLSTM_1/while/add_6AddV2'convLSTM_1/while/convolution_3:output:0'convLSTM_1/while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/add_6y
convLSTM_1/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/while/Const_6y
convLSTM_1/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/while/Const_7±
convLSTM_1/while/Mul_4MulconvLSTM_1/while/add_6:z:0!convLSTM_1/while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Mul_4±
convLSTM_1/while/Add_7AddconvLSTM_1/while/Mul_4:z:0!convLSTM_1/while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Add_7
*convLSTM_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*convLSTM_1/while/clip_by_value_2/Minimum/yë
(convLSTM_1/while/clip_by_value_2/MinimumMinimumconvLSTM_1/while/Add_7:z:03convLSTM_1/while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2*
(convLSTM_1/while/clip_by_value_2/Minimum
"convLSTM_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"convLSTM_1/while/clip_by_value_2/yå
 convLSTM_1/while/clip_by_value_2Maximum,convLSTM_1/while/clip_by_value_2/Minimum:z:0+convLSTM_1/while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2"
 convLSTM_1/while/clip_by_value_2
convLSTM_1/while/Tanh_1TanhconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Tanh_1µ
convLSTM_1/while/mul_5Mul$convLSTM_1/while/clip_by_value_2:z:0convLSTM_1/while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/mul_5
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
convLSTM_1/while/add_8/y
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
convLSTM_1/while/add_9/y­
convLSTM_1/while/add_9AddV2.convlstm_1_while_convlstm_1_while_loop_counter!convLSTM_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2
convLSTM_1/while/add_9
convLSTM_1/while/IdentityIdentityconvLSTM_1/while/add_9:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity
convLSTM_1/while/Identity_1Identity4convlstm_1_while_convlstm_1_while_maximum_iterations*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_1
convLSTM_1/while/Identity_2IdentityconvLSTM_1/while/add_8:z:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_2®
convLSTM_1/while/Identity_3IdentityEconvLSTM_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
convLSTM_1/while/Identity_3
convLSTM_1/while/Identity_4IdentityconvLSTM_1/while/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Identity_4
convLSTM_1/while/Identity_5IdentityconvLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/while/Identity_5"X
)convlstm_1_while_convlstm_1_strided_slice+convlstm_1_while_convlstm_1_strided_slice_0"?
convlstm_1_while_identity"convLSTM_1/while/Identity:output:0"C
convlstm_1_while_identity_1$convLSTM_1/while/Identity_1:output:0"C
convlstm_1_while_identity_2$convLSTM_1/while/Identity_2:output:0"C
convlstm_1_while_identity_3$convLSTM_1/while/Identity_3:output:0"C
convlstm_1_while_identity_4$convLSTM_1/while/Identity_4:output:0"C
convlstm_1_while_identity_5$convLSTM_1/while/Identity_5:output:0"f
0convlstm_1_while_split_1_readvariableop_resource2convlstm_1_while_split_1_readvariableop_resource_0"b
.convlstm_1_while_split_readvariableop_resource0convlstm_1_while_split_readvariableop_resource_0"Ô
gconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensoriconvlstm_1_while_tensorarrayv2read_tensorlistgetitem_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
´

H__inference_functional_1_layer_call_and_return_conditional_losses_577660
input_1
convlstm_1_577638
convlstm_1_577640
batchnorm_1_577643
batchnorm_1_577645
batchnorm_1_577647
batchnorm_1_577649
dense_577653
dense_577655
identity¢#batchnorm_1/StatefulPartitionedCall¢"convLSTM_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall«
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinput_1convlstm_1_577638convlstm_1_577640*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5775122$
"convLSTM_1/StatefulPartitionedCallþ
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0batchnorm_1_577643batchnorm_1_577645batchnorm_1_577647batchnorm_1_577649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5775692%
#batchnorm_1/StatefulPartitionedCall¡
$global_max_pooling2d/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_5771012&
$global_max_pooling2d/PartitionedCall¯
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_577653dense_577655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5776172
dense/StatefulPartitionedCall
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constå
IdentityIdentity&dense/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall#^convLSTM_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
!
_user_specified_name	input_1
±

H__inference_functional_1_layer_call_and_return_conditional_losses_577734

inputs
convlstm_1_577712
convlstm_1_577714
batchnorm_1_577717
batchnorm_1_577719
batchnorm_1_577721
batchnorm_1_577723
dense_577727
dense_577729
identity¢#batchnorm_1/StatefulPartitionedCall¢"convLSTM_1/StatefulPartitionedCall¢dense/StatefulPartitionedCallª
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinputsconvlstm_1_577712convlstm_1_577714*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5775122$
"convLSTM_1/StatefulPartitionedCallþ
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0batchnorm_1_577717batchnorm_1_577719batchnorm_1_577721batchnorm_1_577723*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5775692%
#batchnorm_1/StatefulPartitionedCall¡
$global_max_pooling2d/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_5771012&
$global_max_pooling2d/PartitionedCall¯
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_577727dense_577729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5776172
dense/StatefulPartitionedCall
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constå
IdentityIdentity&dense/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall#^convLSTM_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
³
»
)functional_1_convLSTM_1_while_cond_576279L
Hfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_loop_counterR
Nfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_maximum_iterations-
)functional_1_convlstm_1_while_placeholder/
+functional_1_convlstm_1_while_placeholder_1/
+functional_1_convlstm_1_while_placeholder_2/
+functional_1_convlstm_1_while_placeholder_3L
Hfunctional_1_convlstm_1_while_less_functional_1_convlstm_1_strided_sliced
`functional_1_convlstm_1_while_functional_1_convlstm_1_while_cond_576279___redundant_placeholder0d
`functional_1_convlstm_1_while_functional_1_convlstm_1_while_cond_576279___redundant_placeholder1d
`functional_1_convlstm_1_while_functional_1_convlstm_1_while_cond_576279___redundant_placeholder2*
&functional_1_convlstm_1_while_identity
æ
"functional_1/convLSTM_1/while/LessLess)functional_1_convlstm_1_while_placeholderHfunctional_1_convlstm_1_while_less_functional_1_convlstm_1_strided_slice*
T0*
_output_shapes
: 2$
"functional_1/convLSTM_1/while/Less¥
&functional_1/convLSTM_1/while/IdentityIdentity&functional_1/convLSTM_1/while/Less:z:0*
T0
*
_output_shapes
: 2(
&functional_1/convLSTM_1/while/Identity"Y
&functional_1_convlstm_1_while_identity/functional_1/convLSTM_1/while/Identity:output:0*a
_input_shapesP
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
¾

H__inference_functional_1_layer_call_and_return_conditional_losses_578009

inputs,
(convlstm_1_split_readvariableop_resource.
*convlstm_1_split_1_readvariableop_resource
batchnorm_1_scale
batchnorm_1_offset8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity¢batchnorm_1/AssignNewValue¢batchnorm_1/AssignNewValue_1¢convLSTM_1/while
convLSTM_1/zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
convLSTM_1/zeros_like
 convLSTM_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2"
 convLSTM_1/Sum/reduction_indices¨
convLSTM_1/SumSumconvLSTM_1/zeros_like:y:0)convLSTM_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 2
convLSTM_1/Sum
convLSTM_1/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
convLSTM_1/zerosÒ
convLSTM_1/convolutionConv2DconvLSTM_1/Sum:output:0convLSTM_1/zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution
convLSTM_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose/perm¤
convLSTM_1/transpose	Transposeinputs"convLSTM_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
convLSTM_1/transposel
convLSTM_1/ShapeShapeconvLSTM_1/transpose:y:0*
T0*
_output_shapes
:2
convLSTM_1/Shape
convLSTM_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
convLSTM_1/strided_slice/stack
 convLSTM_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_1
 convLSTM_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 convLSTM_1/strided_slice/stack_2¤
convLSTM_1/strided_sliceStridedSliceconvLSTM_1/Shape:output:0'convLSTM_1/strided_slice/stack:output:0)convLSTM_1/strided_slice/stack_1:output:0)convLSTM_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
convLSTM_1/strided_slice
&convLSTM_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2(
&convLSTM_1/TensorArrayV2/element_shapeÜ
convLSTM_1/TensorArrayV2TensorListReserve/convLSTM_1/TensorArrayV2/element_shape:output:0!convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
convLSTM_1/TensorArrayV2Ý
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          2B
@convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape¤
2convLSTM_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorconvLSTM_1/transpose:y:0IconvLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type024
2convLSTM_1/TensorArrayUnstack/TensorListFromTensor
 convLSTM_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 convLSTM_1/strided_slice_1/stack
"convLSTM_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_1
"convLSTM_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_1/stack_2Ç
convLSTM_1/strided_slice_1StridedSliceconvLSTM_1/transpose:y:0)convLSTM_1/strided_slice_1/stack:output:0+convLSTM_1/strided_slice_1/stack_1:output:0+convLSTM_1/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
convLSTM_1/split/split_dim³
convLSTM_1/split/ReadVariableOpReadVariableOp(convlstm_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype02!
convLSTM_1/split/ReadVariableOpó
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
convLSTM_1/split_1/split_dim¹
!convLSTM_1/split_1/ReadVariableOpReadVariableOp*convlstm_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!convLSTM_1/split_1/ReadVariableOpû
convLSTM_1/split_1Split%convLSTM_1/split_1/split_dim:output:0)convLSTM_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
convLSTM_1/split_1â
convLSTM_1/convolution_1Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_1â
convLSTM_1/convolution_2Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_2â
convLSTM_1/convolution_3Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_3â
convLSTM_1/convolution_4Conv2D#convLSTM_1/strided_slice_1:output:0convLSTM_1/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convLSTM_1/convolution_4ß
convLSTM_1/convolution_5Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_5ß
convLSTM_1/convolution_6Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_6ß
convLSTM_1/convolution_7Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_7ß
convLSTM_1/convolution_8Conv2DconvLSTM_1/convolution:output:0convLSTM_1/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convLSTM_1/convolution_8ª
convLSTM_1/addAddV2!convLSTM_1/convolution_1:output:0!convLSTM_1/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/addm
convLSTM_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/Const_2m
convLSTM_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_3
convLSTM_1/MulMulconvLSTM_1/add:z:0convLSTM_1/Const_2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Mul
convLSTM_1/Add_1AddconvLSTM_1/Mul:z:0convLSTM_1/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Add_1
"convLSTM_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"convLSTM_1/clip_by_value/Minimum/yÍ
 convLSTM_1/clip_by_value/MinimumMinimumconvLSTM_1/Add_1:z:0+convLSTM_1/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2"
 convLSTM_1/clip_by_value/Minimum}
convLSTM_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value/yÅ
convLSTM_1/clip_by_valueMaximum$convLSTM_1/clip_by_value/Minimum:z:0#convLSTM_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/clip_by_value®
convLSTM_1/add_2AddV2!convLSTM_1/convolution_2:output:0!convLSTM_1/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_2m
convLSTM_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/Const_4m
convLSTM_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_5
convLSTM_1/Mul_1MulconvLSTM_1/add_2:z:0convLSTM_1/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Mul_1
convLSTM_1/Add_3AddconvLSTM_1/Mul_1:z:0convLSTM_1/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Add_3
$convLSTM_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$convLSTM_1/clip_by_value_1/Minimum/yÓ
"convLSTM_1/clip_by_value_1/MinimumMinimumconvLSTM_1/Add_3:z:0-convLSTM_1/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2$
"convLSTM_1/clip_by_value_1/Minimum
convLSTM_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_1/yÍ
convLSTM_1/clip_by_value_1Maximum&convLSTM_1/clip_by_value_1/Minimum:z:0%convLSTM_1/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/clip_by_value_1§
convLSTM_1/mul_2MulconvLSTM_1/clip_by_value_1:z:0convLSTM_1/convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/mul_2®
convLSTM_1/add_4AddV2!convLSTM_1/convolution_3:output:0!convLSTM_1/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_4{
convLSTM_1/TanhTanhconvLSTM_1/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Tanh
convLSTM_1/mul_3MulconvLSTM_1/clip_by_value:z:0convLSTM_1/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/mul_3
convLSTM_1/add_5AddV2convLSTM_1/mul_2:z:0convLSTM_1/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_5®
convLSTM_1/add_6AddV2!convLSTM_1/convolution_4:output:0!convLSTM_1/convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/add_6m
convLSTM_1/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
convLSTM_1/Const_6m
convLSTM_1/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
convLSTM_1/Const_7
convLSTM_1/Mul_4MulconvLSTM_1/add_6:z:0convLSTM_1/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Mul_4
convLSTM_1/Add_7AddconvLSTM_1/Mul_4:z:0convLSTM_1/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Add_7
$convLSTM_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$convLSTM_1/clip_by_value_2/Minimum/yÓ
"convLSTM_1/clip_by_value_2/MinimumMinimumconvLSTM_1/Add_7:z:0-convLSTM_1/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2$
"convLSTM_1/clip_by_value_2/Minimum
convLSTM_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
convLSTM_1/clip_by_value_2/yÍ
convLSTM_1/clip_by_value_2Maximum&convLSTM_1/clip_by_value_2/Minimum:z:0%convLSTM_1/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/clip_by_value_2
convLSTM_1/Tanh_1TanhconvLSTM_1/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/Tanh_1
convLSTM_1/mul_5MulconvLSTM_1/clip_by_value_2:z:0convLSTM_1/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/mul_5­
(convLSTM_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2*
(convLSTM_1/TensorArrayV2_1/element_shapeâ
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
convLSTM_1/time
#convLSTM_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#convLSTM_1/while/maximum_iterations
convLSTM_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
convLSTM_1/while/loop_counterä
convLSTM_1/whileWhile&convLSTM_1/while/loop_counter:output:0,convLSTM_1/while/maximum_iterations:output:0convLSTM_1/time:output:0#convLSTM_1/TensorArrayV2_1:handle:0convLSTM_1/convolution:output:0convLSTM_1/convolution:output:0!convLSTM_1/strided_slice:output:0BconvLSTM_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0(convlstm_1_split_readvariableop_resource*convlstm_1_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*(
body R
convLSTM_1_while_body_577870*(
cond R
convLSTM_1_while_cond_577869*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
convLSTM_1/whileÓ
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2=
;convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape
-convLSTM_1/TensorArrayV2Stack/TensorListStackTensorListStackconvLSTM_1/while:output:3DconvLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02/
-convLSTM_1/TensorArrayV2Stack/TensorListStack
 convLSTM_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2"
 convLSTM_1/strided_slice_2/stack
"convLSTM_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"convLSTM_1/strided_slice_2/stack_1
"convLSTM_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"convLSTM_1/strided_slice_2/stack_2å
convLSTM_1/strided_slice_2StridedSlice6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0)convLSTM_1/strided_slice_2/stack:output:0+convLSTM_1/strided_slice_2/stack_1:output:0+convLSTM_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
convLSTM_1/strided_slice_2
convLSTM_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
convLSTM_1/transpose_1/permÚ
convLSTM_1/transpose_1	Transpose6convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:0$convLSTM_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2
convLSTM_1/transpose_1Ë
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1
batchnorm_1/FusedBatchNormV3FusedBatchNormV3#convLSTM_1/strided_slice_2:output:0batchnorm_1_scalebatchnorm_1_offset3batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_1/FusedBatchNormV3Ç
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValueÕ
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue_1©
*global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2,
*global_max_pooling2d/Max/reduction_indicesÄ
global_max_pooling2d/MaxMax batchnorm_1/FusedBatchNormV3:y:03global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
global_max_pooling2d/Max
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp 
dense/MatMulMatMul!global_max_pooling2d/Max:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Sigmoid
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const´
IdentityIdentitydense/Sigmoid:y:0^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1^convLSTM_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::28
batchnorm_1/AssignNewValuebatchnorm_1/AssignNewValue2<
batchnorm_1/AssignNewValue_1batchnorm_1/AssignNewValue_12$
convLSTM_1/whileconvLSTM_1/while:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
èe
µ
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_577512

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity¢whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1¶
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¶
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¶
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¶
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_4³
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5³
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6³
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7³
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_577396*
condR
while_cond_577395*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const}
IdentityIdentitystrided_slice_2:output:0^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿx ::2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
Ç
Ý
-__inference_functional_1_layer_call_fn_577753
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_5777342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
!
_user_specified_name	input_1
Å
ÿ
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579132

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¶
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿv::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs


while_cond_578559
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_578559___redundant_placeholder04
0while_while_cond_578559___redundant_placeholder14
0while_while_cond_578559___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
Ù	
Û
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_577083

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¹
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ø

)functional_1_convLSTM_1_while_body_576280L
Hfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_loop_counterR
Nfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_maximum_iterations-
)functional_1_convlstm_1_while_placeholder/
+functional_1_convlstm_1_while_placeholder_1/
+functional_1_convlstm_1_while_placeholder_2/
+functional_1_convlstm_1_while_placeholder_3I
Efunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_slice_0
functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0A
=functional_1_convlstm_1_while_split_readvariableop_resource_0C
?functional_1_convlstm_1_while_split_1_readvariableop_resource_0*
&functional_1_convlstm_1_while_identity,
(functional_1_convlstm_1_while_identity_1,
(functional_1_convlstm_1_while_identity_2,
(functional_1_convlstm_1_while_identity_3,
(functional_1_convlstm_1_while_identity_4,
(functional_1_convlstm_1_while_identity_5G
Cfunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_slice
functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor?
;functional_1_convlstm_1_while_split_readvariableop_resourceA
=functional_1_convlstm_1_while_split_1_readvariableop_resourceû
Ofunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          2Q
Ofunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shapeí
Afunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemfunctional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0)functional_1_convlstm_1_while_placeholderXfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
element_dtype02C
Afunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem
#functional_1/convLSTM_1/while/ConstConst*
_output_shapes
: *
dtype0*
value	B :2%
#functional_1/convLSTM_1/while/Const 
-functional_1/convLSTM_1/while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/convLSTM_1/while/split/split_dimî
2functional_1/convLSTM_1/while/split/ReadVariableOpReadVariableOp=functional_1_convlstm_1_while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype024
2functional_1/convLSTM_1/while/split/ReadVariableOp¿
#functional_1/convLSTM_1/while/splitSplit6functional_1/convLSTM_1/while/split/split_dim:output:0:functional_1/convLSTM_1/while/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2%
#functional_1/convLSTM_1/while/split
%functional_1/convLSTM_1/while/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_1/while/Const_1¤
/functional_1/convLSTM_1/while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/functional_1/convLSTM_1/while/split_1/split_dimô
4functional_1/convLSTM_1/while/split_1/ReadVariableOpReadVariableOp?functional_1_convlstm_1_while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype026
4functional_1/convLSTM_1/while/split_1/ReadVariableOpÇ
%functional_1/convLSTM_1/while/split_1Split8functional_1/convLSTM_1/while/split_1/split_dim:output:0<functional_1/convLSTM_1/while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2'
%functional_1/convLSTM_1/while/split_1¼
)functional_1/convLSTM_1/while/convolutionConv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2+
)functional_1/convLSTM_1/while/convolutionÀ
+functional_1/convLSTM_1/while/convolution_1Conv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2-
+functional_1/convLSTM_1/while/convolution_1À
+functional_1/convLSTM_1/while/convolution_2Conv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2-
+functional_1/convLSTM_1/while/convolution_2À
+functional_1/convLSTM_1/while/convolution_3Conv2DHfunctional_1/convLSTM_1/while/TensorArrayV2Read/TensorListGetItem:item:0,functional_1/convLSTM_1/while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2-
+functional_1/convLSTM_1/while/convolution_3¤
+functional_1/convLSTM_1/while/convolution_4Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_4¤
+functional_1/convLSTM_1/while/convolution_5Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_5¤
+functional_1/convLSTM_1/while/convolution_6Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_6¤
+functional_1/convLSTM_1/while/convolution_7Conv2D+functional_1_convlstm_1_while_placeholder_2.functional_1/convLSTM_1/while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2-
+functional_1/convLSTM_1/while/convolution_7ô
!functional_1/convLSTM_1/while/addAddV22functional_1/convLSTM_1/while/convolution:output:04functional_1/convLSTM_1/while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2#
!functional_1/convLSTM_1/while/add
%functional_1/convLSTM_1/while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2'
%functional_1/convLSTM_1/while/Const_2
%functional_1/convLSTM_1/while/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_1/while/Const_3ß
!functional_1/convLSTM_1/while/MulMul%functional_1/convLSTM_1/while/add:z:0.functional_1/convLSTM_1/while/Const_2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2#
!functional_1/convLSTM_1/while/Mulã
#functional_1/convLSTM_1/while/Add_1Add%functional_1/convLSTM_1/while/Mul:z:0.functional_1/convLSTM_1/while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/Add_1³
5functional_1/convLSTM_1/while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5functional_1/convLSTM_1/while/clip_by_value/Minimum/y
3functional_1/convLSTM_1/while/clip_by_value/MinimumMinimum'functional_1/convLSTM_1/while/Add_1:z:0>functional_1/convLSTM_1/while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv25
3functional_1/convLSTM_1/while/clip_by_value/Minimum£
-functional_1/convLSTM_1/while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-functional_1/convLSTM_1/while/clip_by_value/y
+functional_1/convLSTM_1/while/clip_by_valueMaximum7functional_1/convLSTM_1/while/clip_by_value/Minimum:z:06functional_1/convLSTM_1/while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2-
+functional_1/convLSTM_1/while/clip_by_valueú
#functional_1/convLSTM_1/while/add_2AddV24functional_1/convLSTM_1/while/convolution_1:output:04functional_1/convLSTM_1/while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/add_2
%functional_1/convLSTM_1/while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2'
%functional_1/convLSTM_1/while/Const_4
%functional_1/convLSTM_1/while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_1/while/Const_5å
#functional_1/convLSTM_1/while/Mul_1Mul'functional_1/convLSTM_1/while/add_2:z:0.functional_1/convLSTM_1/while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/Mul_1å
#functional_1/convLSTM_1/while/Add_3Add'functional_1/convLSTM_1/while/Mul_1:z:0.functional_1/convLSTM_1/while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/Add_3·
7functional_1/convLSTM_1/while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7functional_1/convLSTM_1/while/clip_by_value_1/Minimum/y
5functional_1/convLSTM_1/while/clip_by_value_1/MinimumMinimum'functional_1/convLSTM_1/while/Add_3:z:0@functional_1/convLSTM_1/while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv27
5functional_1/convLSTM_1/while/clip_by_value_1/Minimum§
/functional_1/convLSTM_1/while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/functional_1/convLSTM_1/while/clip_by_value_1/y
-functional_1/convLSTM_1/while/clip_by_value_1Maximum9functional_1/convLSTM_1/while/clip_by_value_1/Minimum:z:08functional_1/convLSTM_1/while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2/
-functional_1/convLSTM_1/while/clip_by_value_1ì
#functional_1/convLSTM_1/while/mul_2Mul1functional_1/convLSTM_1/while/clip_by_value_1:z:0+functional_1_convlstm_1_while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/mul_2ú
#functional_1/convLSTM_1/while/add_4AddV24functional_1/convLSTM_1/while/convolution_2:output:04functional_1/convLSTM_1/while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/add_4´
"functional_1/convLSTM_1/while/TanhTanh'functional_1/convLSTM_1/while/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2$
"functional_1/convLSTM_1/while/Tanhå
#functional_1/convLSTM_1/while/mul_3Mul/functional_1/convLSTM_1/while/clip_by_value:z:0&functional_1/convLSTM_1/while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/mul_3à
#functional_1/convLSTM_1/while/add_5AddV2'functional_1/convLSTM_1/while/mul_2:z:0'functional_1/convLSTM_1/while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/add_5ú
#functional_1/convLSTM_1/while/add_6AddV24functional_1/convLSTM_1/while/convolution_3:output:04functional_1/convLSTM_1/while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/add_6
%functional_1/convLSTM_1/while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2'
%functional_1/convLSTM_1/while/Const_6
%functional_1/convLSTM_1/while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%functional_1/convLSTM_1/while/Const_7å
#functional_1/convLSTM_1/while/Mul_4Mul'functional_1/convLSTM_1/while/add_6:z:0.functional_1/convLSTM_1/while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/Mul_4å
#functional_1/convLSTM_1/while/Add_7Add'functional_1/convLSTM_1/while/Mul_4:z:0.functional_1/convLSTM_1/while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/Add_7·
7functional_1/convLSTM_1/while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7functional_1/convLSTM_1/while/clip_by_value_2/Minimum/y
5functional_1/convLSTM_1/while/clip_by_value_2/MinimumMinimum'functional_1/convLSTM_1/while/Add_7:z:0@functional_1/convLSTM_1/while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv27
5functional_1/convLSTM_1/while/clip_by_value_2/Minimum§
/functional_1/convLSTM_1/while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/functional_1/convLSTM_1/while/clip_by_value_2/y
-functional_1/convLSTM_1/while/clip_by_value_2Maximum9functional_1/convLSTM_1/while/clip_by_value_2/Minimum:z:08functional_1/convLSTM_1/while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2/
-functional_1/convLSTM_1/while/clip_by_value_2¸
$functional_1/convLSTM_1/while/Tanh_1Tanh'functional_1/convLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2&
$functional_1/convLSTM_1/while/Tanh_1é
#functional_1/convLSTM_1/while/mul_5Mul1functional_1/convLSTM_1/while/clip_by_value_2:z:0(functional_1/convLSTM_1/while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/while/mul_5Ë
Bfunctional_1/convLSTM_1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+functional_1_convlstm_1_while_placeholder_1)functional_1_convlstm_1_while_placeholder'functional_1/convLSTM_1/while/mul_5:z:0*
_output_shapes
: *
element_dtype02D
Bfunctional_1/convLSTM_1/while/TensorArrayV2Write/TensorListSetItem
%functional_1/convLSTM_1/while/add_8/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_1/while/add_8/yÏ
#functional_1/convLSTM_1/while/add_8AddV2)functional_1_convlstm_1_while_placeholder.functional_1/convLSTM_1/while/add_8/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/convLSTM_1/while/add_8
%functional_1/convLSTM_1/while/add_9/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%functional_1/convLSTM_1/while/add_9/yî
#functional_1/convLSTM_1/while/add_9AddV2Hfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_loop_counter.functional_1/convLSTM_1/while/add_9/y:output:0*
T0*
_output_shapes
: 2%
#functional_1/convLSTM_1/while/add_9¦
&functional_1/convLSTM_1/while/IdentityIdentity'functional_1/convLSTM_1/while/add_9:z:0*
T0*
_output_shapes
: 2(
&functional_1/convLSTM_1/while/IdentityÑ
(functional_1/convLSTM_1/while/Identity_1IdentityNfunctional_1_convlstm_1_while_functional_1_convlstm_1_while_maximum_iterations*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_1/while/Identity_1ª
(functional_1/convLSTM_1/while/Identity_2Identity'functional_1/convLSTM_1/while/add_8:z:0*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_1/while/Identity_2Õ
(functional_1/convLSTM_1/while/Identity_3IdentityRfunctional_1/convLSTM_1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2*
(functional_1/convLSTM_1/while/Identity_3Ä
(functional_1/convLSTM_1/while/Identity_4Identity'functional_1/convLSTM_1/while/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2*
(functional_1/convLSTM_1/while/Identity_4Ä
(functional_1/convLSTM_1/while/Identity_5Identity'functional_1/convLSTM_1/while/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2*
(functional_1/convLSTM_1/while/Identity_5"
Cfunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_sliceEfunctional_1_convlstm_1_while_functional_1_convlstm_1_strided_slice_0"Y
&functional_1_convlstm_1_while_identity/functional_1/convLSTM_1/while/Identity:output:0"]
(functional_1_convlstm_1_while_identity_11functional_1/convLSTM_1/while/Identity_1:output:0"]
(functional_1_convlstm_1_while_identity_21functional_1/convLSTM_1/while/Identity_2:output:0"]
(functional_1_convlstm_1_while_identity_31functional_1/convLSTM_1/while/Identity_3:output:0"]
(functional_1_convlstm_1_while_identity_41functional_1/convLSTM_1/while/Identity_4:output:0"]
(functional_1_convlstm_1_while_identity_51functional_1/convLSTM_1/while/Identity_5:output:0"
=functional_1_convlstm_1_while_split_1_readvariableop_resource?functional_1_convlstm_1_while_split_1_readvariableop_resource_0"|
;functional_1_convlstm_1_while_split_readvariableop_resource=functional_1_convlstm_1_while_split_readvariableop_resource_0"
functional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensorfunctional_1_convlstm_1_while_tensorarrayv2read_tensorlistgetitem_functional_1_convlstm_1_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
ã;
î
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_576574

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
identity

identity_1

identity_2P
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1 
convolutionConv2Dinputssplit:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution¤
convolution_1Conv2Dinputssplit:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¤
convolution_2Conv2Dinputssplit:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¤
convolution_3Conv2Dinputssplit:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¥
convolution_4Conv2Dstatessplit_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_4¥
convolution_5Conv2Dstatessplit_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5¥
convolution_6Conv2Dstatessplit_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6¥
convolution_7Conv2Dstatessplit_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7|
addAddV2convolution:output:0convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1o
mul_2Mulclip_by_value_1:z:0states_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
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
:ÿÿÿÿÿÿÿÿÿv2

Identityj

Identity_1Identity	mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_1j

Identity_2Identity	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿx :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_namestates:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_namestates
ã
,
__inference_loss_fn_0_579424
identity
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


while_cond_577395
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_577395___redundant_placeholder04
0while_while_cond_577395___redundant_placeholder14
0while_while_cond_577395___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
¬X
Ì
while_body_578779
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
%while_split_1_readvariableop_resourceË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
while/split/split_dim¦
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOpß
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
while/split_1/split_dim¬
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOpç
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1Ü
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolutionà
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_1à
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_2à
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_3Ä
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_4Ä
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_5Ä
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_6Ä
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_7
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
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
:ÿÿÿÿÿÿÿÿÿv2
	while/Mul
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_1
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/clip_by_value/Minimum/y¹
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y±
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_1
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_3
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_1/Minimum/y¿
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y¹
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_2
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

while/Tanh
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_5
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_4
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_7
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_2/Minimum/y¿
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y¹
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Tanh_1
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_5Ó
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
°

H__inference_functional_1_layer_call_and_return_conditional_losses_577635
input_1
convlstm_1_577531
convlstm_1_577533
batchnorm_1_577596
batchnorm_1_577598
batchnorm_1_577600
batchnorm_1_577602
dense_577628
dense_577630
identity¢#batchnorm_1/StatefulPartitionedCall¢"convLSTM_1/StatefulPartitionedCall¢dense/StatefulPartitionedCall«
"convLSTM_1/StatefulPartitionedCallStatefulPartitionedCallinput_1convlstm_1_577531convlstm_1_577533*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5773112$
"convLSTM_1/StatefulPartitionedCallú
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall+convLSTM_1/StatefulPartitionedCall:output:0batchnorm_1_577596batchnorm_1_577598batchnorm_1_577600batchnorm_1_577602*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5775532%
#batchnorm_1/StatefulPartitionedCall¡
$global_max_pooling2d/PartitionedCallPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_5771012&
$global_max_pooling2d/PartitionedCall¯
dense/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling2d/PartitionedCall:output:0dense_577628dense_577630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5776172
dense/StatefulPartitionedCall
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Constå
IdentityIdentity&dense/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall#^convLSTM_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2H
"convLSTM_1/StatefulPartitionedCall"convLSTM_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
!
_user_specified_name	input_1
¦f
·
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_579096
inputs_0!
split_readvariableop_resource#
split_1_readvariableop_resource
identity¢whilew

zeros_like	ZerosLikeinputs_0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1¶
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¶
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¶
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¶
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_4³
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5³
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6³
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7³
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_578980*
condR
while_cond_578979*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeú
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm·
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const}
IdentityIdentitystrided_slice_2:output:0^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx ::2
whilewhile:g c
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 
"
_user_specified_name
inputs/0
¬X
Ì
while_body_577396
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
%while_split_1_readvariableop_resourceË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
while/split/split_dim¦
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOpß
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
while/split_1/split_dim¬
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOpç
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1Ü
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolutionà
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_1à
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_2à
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_3Ä
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_4Ä
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_5Ä
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_6Ä
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_7
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
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
:ÿÿÿÿÿÿÿÿÿv2
	while/Mul
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_1
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/clip_by_value/Minimum/y¹
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y±
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_1
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_3
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_1/Minimum/y¿
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y¹
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_2
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

while/Tanh
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_5
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_4
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_7
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_2/Minimum/y¿
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y¹
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Tanh_1
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_5Ó
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
èe
µ
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578676

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity¢whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1¶
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¶
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¶
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¶
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_4³
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5³
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6³
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7³
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_578560*
condR
while_cond_578559*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const}
IdentityIdentitystrided_slice_2:output:0^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿx ::2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
¬X
Ì
while_body_578980
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
%while_split_1_readvariableop_resourceË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
while/split/split_dim¦
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOpß
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
while/split_1/split_dim¬
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOpç
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1Ü
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolutionà
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_1à
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_2à
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_3Ä
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_4Ä
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_5Ä
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_6Ä
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_7
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
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
:ÿÿÿÿÿÿÿÿÿv2
	while/Mul
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_1
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/clip_by_value/Minimum/y¹
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y±
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_1
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_3
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_1/Minimum/y¿
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y¹
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_2
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

while/Tanh
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_5
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_4
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_7
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_2/Minimum/y¿
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y¹
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Tanh_1
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_5Ó
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
¬X
Ì
while_body_578359
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
%while_split_1_readvariableop_resourceË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
while/split/split_dim¦
while/split/ReadVariableOpReadVariableOp%while_split_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split/ReadVariableOpß
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
while/split_1/split_dim¬
while/split_1/ReadVariableOpReadVariableOp'while_split_1_readvariableop_resource_0*&
_output_shapes
:@*
dtype02
while/split_1/ReadVariableOpç
while/split_1Split while/split_1/split_dim:output:0$while/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
while/split_1Ü
while/convolutionConv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolutionà
while/convolution_1Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_1à
while/convolution_2Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_2à
while/convolution_3Conv2D0while/TensorArrayV2Read/TensorListGetItem:item:0while/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
while/convolution_3Ä
while/convolution_4Conv2Dwhile_placeholder_2while/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_4Ä
while/convolution_5Conv2Dwhile_placeholder_2while/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_5Ä
while/convolution_6Conv2Dwhile_placeholder_2while/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_6Ä
while/convolution_7Conv2Dwhile_placeholder_2while/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
while/convolution_7
	while/addAddV2while/convolution:output:0while/convolution_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
	while/addc
while/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
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
:ÿÿÿÿÿÿÿÿÿv2
	while/Mul
while/Add_1Addwhile/Mul:z:0while/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_1
while/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/clip_by_value/Minimum/y¹
while/clip_by_value/MinimumMinimumwhile/Add_1:z:0&while/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value/Minimums
while/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value/y±
while/clip_by_valueMaximumwhile/clip_by_value/Minimum:z:0while/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value
while/add_2AddV2while/convolution_1:output:0while/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_2c
while/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_4c
while/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_5
while/Mul_1Mulwhile/add_2:z:0while/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_1
while/Add_3Addwhile/Mul_1:z:0while/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_3
while/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_1/Minimum/y¿
while/clip_by_value_1/MinimumMinimumwhile/Add_3:z:0(while/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1/Minimumw
while/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_1/y¹
while/clip_by_value_1Maximum!while/clip_by_value_1/Minimum:z:0 while/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_1
while/mul_2Mulwhile/clip_by_value_1:z:0while_placeholder_3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_2
while/add_4AddV2while/convolution_2:output:0while/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_4l

while/TanhTanhwhile/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

while/Tanh
while/mul_3Mulwhile/clip_by_value:z:0while/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_3
while/add_5AddV2while/mul_2:z:0while/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_5
while/add_6AddV2while/convolution_3:output:0while/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/add_6c
while/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/Const_6c
while/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/Const_7
while/Mul_4Mulwhile/add_6:z:0while/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Mul_4
while/Add_7Addwhile/Mul_4:z:0while/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Add_7
while/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
while/clip_by_value_2/Minimum/y¿
while/clip_by_value_2/MinimumMinimumwhile/Add_7:z:0(while/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2/Minimumw
while/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
while/clip_by_value_2/y¹
while/clip_by_value_2Maximum!while/clip_by_value_2/Minimum:z:0 while/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/clip_by_value_2p
while/Tanh_1Tanhwhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Tanh_1
while/mul_5Mulwhile/clip_by_value_2:z:0while/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/mul_5Ó
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3|
while/Identity_4Identitywhile/mul_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4|
while/Identity_5Identitywhile/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"P
%while_split_1_readvariableop_resource'while_split_1_readvariableop_resource_0"L
#while_split_readvariableop_resource%while_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 
ç6
£
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_576883

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall¢whileu

zeros_like	ZerosLikeinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
shrink_axis_mask2
strided_slice_1
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0convolution:output:0convolution:output:0unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_5765072
StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0unknown	unknown_0*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_576818*
condR
while_cond_576817*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeú
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm·
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const
IdentityIdentitystrided_slice_2:output:0^StatefulPartitionedCall^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx ::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs

Ô
$__inference_signature_wrapper_577785
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_5764162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
!
_user_specified_name	input_1

Ã
2__inference_conv_lst_m2d_cell_layer_call_fn_579404

inputs
states_0
states_1
unknown
	unknown_0
identity

identity_1

identity_2¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_5765072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*o
_input_shapes^
\:ÿÿÿÿÿÿÿÿÿx :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
"
_user_specified_name
states/1


+__inference_convLSTM_1_layer_call_fn_578685

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_5773112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿx ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
È

,__inference_batchnorm_1_layer_call_fn_579161

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_5775532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿv::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
èe
µ
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_577311

inputs!
split_readvariableop_resource#
split_1_readvariableop_resource
identity¢whilel

zeros_like	ZerosLikeinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2

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
:ÿÿÿÿÿÿÿÿÿx 2
Sums
zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
zeros¦
convolutionConv2DSum:output:0zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2Ç
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_1/stack_2
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
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
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*&
_output_shapes
:@*
dtype02
split/ReadVariableOpÇ
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
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*&
_output_shapes
:@*
dtype02
split_1/ReadVariableOpÏ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2	
split_1¶
convolution_1Conv2Dstrided_slice_1:output:0split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_1¶
convolution_2Conv2Dstrided_slice_1:output:0split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_2¶
convolution_3Conv2Dstrided_slice_1:output:0split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_3¶
convolution_4Conv2Dstrided_slice_1:output:0split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2
convolution_4³
convolution_5Conv2Dconvolution:output:0split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_5³
convolution_6Conv2Dconvolution:output:0split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_6³
convolution_7Conv2Dconvolution:output:0split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_7³
convolution_8Conv2Dconvolution:output:0split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2
convolution_8~
addAddV2convolution_1:output:0convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mulk
Add_1AddMul:z:0Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y¡
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value
add_2AddV2convolution_2:output:0convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_1m
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y§
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y¡
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_1{
mul_2Mulclip_by_value_1:z:0convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_2
add_4AddV2convolution_3:output:0convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_4Z
TanhTanh	add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanhm
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_3h
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_5
add_6AddV2convolution_4:output:0convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
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
:ÿÿÿÿÿÿÿÿÿv2
Mul_4m
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y§
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y¡
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
clip_by_value_2^
Tanh_1Tanh	add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
Tanh_1q
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2
TensorArrayV2_1/element_shape¶
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0convolution:output:0convolution:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0split_readvariableop_resourcesplit_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*
bodyR
while_body_577195*
condR
while_cond_577194*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
while½
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_2/stack_2£
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2
strided_slice_2
transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2
transpose_1
#convLSTM_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#convLSTM_1/kernel/Regularizer/Const}
IdentityIdentitystrided_slice_2:output:0^while*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿx ::2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
 
_user_specified_nameinputs
Ù	
Û
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579208

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¹
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*T
_input_shapesC
A:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


while_cond_577194
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_577194___redundant_placeholder04
0while_while_cond_577194___redundant_placeholder14
0while_while_cond_577194___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:


while_cond_578979
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice4
0while_while_cond_578979___redundant_placeholder04
0while_while_cond_578979___redundant_placeholder14
0while_while_cond_578979___redundant_placeholder2
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
N: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: :::: 
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
:
§"

while_body_576818
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_576842_0
while_576844_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_576842
while_576844¢while/StatefulPartitionedCallË
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÜ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem±
while/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_576842_0while_576844_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *h
_output_shapesV
T:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_5765072
while/StatefulPartitionedCallê
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
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2­
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3³
while/Identity_4Identity&while/StatefulPartitionedCall:output:1^while/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_4³
while/Identity_5Identity&while/StatefulPartitionedCall:output:2^while/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
while/Identity_5"
while_576842while_576842_0"
while_576844while_576844_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*_
_input_shapesN
L: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : ::2>
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
:ÿÿÿÿÿÿÿÿÿv:62
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv:

_output_shapes
: :

_output_shapes
: 

Q
5__inference_global_max_pooling2d_layer_call_fn_577107

inputs
identityÚ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_5771012
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579148

inputs	
scale

offset,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1¨
FusedBatchNormV3FusedBatchNormV3inputsscaleoffset'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:ÿÿÿÿÿÿÿÿÿv:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
 
_user_specified_nameinputs
Å®
¡
!__inference__wrapped_model_576416
input_19
5functional_1_convlstm_1_split_readvariableop_resource;
7functional_1_convlstm_1_split_1_readvariableop_resource"
functional_1_batchnorm_1_scale#
functional_1_batchnorm_1_offsetE
Afunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource5
1functional_1_dense_matmul_readvariableop_resource6
2functional_1_dense_biasadd_readvariableop_resource
identity¢functional_1/convLSTM_1/while
"functional_1/convLSTM_1/zeros_like	ZerosLikeinput_1*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2$
"functional_1/convLSTM_1/zeros_like 
-functional_1/convLSTM_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-functional_1/convLSTM_1/Sum/reduction_indicesÜ
functional_1/convLSTM_1/SumSum&functional_1/convLSTM_1/zeros_like:y:06functional_1/convLSTM_1/Sum/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx 2
functional_1/convLSTM_1/Sum£
functional_1/convLSTM_1/zerosConst*&
_output_shapes
:*
dtype0*%
valueB*    2
functional_1/convLSTM_1/zeros
#functional_1/convLSTM_1/convolutionConv2D$functional_1/convLSTM_1/Sum:output:0&functional_1/convLSTM_1/zeros:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2%
#functional_1/convLSTM_1/convolution­
&functional_1/convLSTM_1/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2(
&functional_1/convLSTM_1/transpose/permÌ
!functional_1/convLSTM_1/transpose	Transposeinput_1/functional_1/convLSTM_1/transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 2#
!functional_1/convLSTM_1/transpose
functional_1/convLSTM_1/ShapeShape%functional_1/convLSTM_1/transpose:y:0*
T0*
_output_shapes
:2
functional_1/convLSTM_1/Shape¤
+functional_1/convLSTM_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/convLSTM_1/strided_slice/stack¨
-functional_1/convLSTM_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/convLSTM_1/strided_slice/stack_1¨
-functional_1/convLSTM_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/convLSTM_1/strided_slice/stack_2ò
%functional_1/convLSTM_1/strided_sliceStridedSlice&functional_1/convLSTM_1/Shape:output:04functional_1/convLSTM_1/strided_slice/stack:output:06functional_1/convLSTM_1/strided_slice/stack_1:output:06functional_1/convLSTM_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%functional_1/convLSTM_1/strided_sliceµ
3functional_1/convLSTM_1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ25
3functional_1/convLSTM_1/TensorArrayV2/element_shape
%functional_1/convLSTM_1/TensorArrayV2TensorListReserve<functional_1/convLSTM_1/TensorArrayV2/element_shape:output:0.functional_1/convLSTM_1/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%functional_1/convLSTM_1/TensorArrayV2÷
Mfunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿx          2O
Mfunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shapeØ
?functional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%functional_1/convLSTM_1/transpose:y:0Vfunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02A
?functional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor¨
-functional_1/convLSTM_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/convLSTM_1/strided_slice_1/stack¬
/functional_1/convLSTM_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_1/strided_slice_1/stack_1¬
/functional_1/convLSTM_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_1/strided_slice_1/stack_2
'functional_1/convLSTM_1/strided_slice_1StridedSlice%functional_1/convLSTM_1/transpose:y:06functional_1/convLSTM_1/strided_slice_1/stack:output:08functional_1/convLSTM_1/strided_slice_1/stack_1:output:08functional_1/convLSTM_1/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿx *
shrink_axis_mask2)
'functional_1/convLSTM_1/strided_slice_1
functional_1/convLSTM_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
functional_1/convLSTM_1/Const
'functional_1/convLSTM_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2)
'functional_1/convLSTM_1/split/split_dimÚ
,functional_1/convLSTM_1/split/ReadVariableOpReadVariableOp5functional_1_convlstm_1_split_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,functional_1/convLSTM_1/split/ReadVariableOp§
functional_1/convLSTM_1/splitSplit0functional_1/convLSTM_1/split/split_dim:output:04functional_1/convLSTM_1/split/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2
functional_1/convLSTM_1/split
functional_1/convLSTM_1/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/convLSTM_1/Const_1
)functional_1/convLSTM_1/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)functional_1/convLSTM_1/split_1/split_dimà
.functional_1/convLSTM_1/split_1/ReadVariableOpReadVariableOp7functional_1_convlstm_1_split_1_readvariableop_resource*&
_output_shapes
:@*
dtype020
.functional_1/convLSTM_1/split_1/ReadVariableOp¯
functional_1/convLSTM_1/split_1Split2functional_1/convLSTM_1/split_1/split_dim:output:06functional_1/convLSTM_1/split_1/ReadVariableOp:value:0*
T0*\
_output_shapesJ
H::::*
	num_split2!
functional_1/convLSTM_1/split_1
%functional_1/convLSTM_1/convolution_1Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_1
%functional_1/convLSTM_1/convolution_2Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_2
%functional_1/convLSTM_1/convolution_3Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_3
%functional_1/convLSTM_1/convolution_4Conv2D0functional_1/convLSTM_1/strided_slice_1:output:0&functional_1/convLSTM_1/split:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingVALID*
strides
2'
%functional_1/convLSTM_1/convolution_4
%functional_1/convLSTM_1/convolution_5Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_5
%functional_1/convLSTM_1/convolution_6Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_6
%functional_1/convLSTM_1/convolution_7Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:2*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_7
%functional_1/convLSTM_1/convolution_8Conv2D,functional_1/convLSTM_1/convolution:output:0(functional_1/convLSTM_1/split_1:output:3*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
paddingSAME*
strides
2'
%functional_1/convLSTM_1/convolution_8Þ
functional_1/convLSTM_1/addAddV2.functional_1/convLSTM_1/convolution_1:output:0.functional_1/convLSTM_1/convolution_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/add
functional_1/convLSTM_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2!
functional_1/convLSTM_1/Const_2
functional_1/convLSTM_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_1/Const_3Ç
functional_1/convLSTM_1/MulMulfunctional_1/convLSTM_1/add:z:0(functional_1/convLSTM_1/Const_2:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/MulË
functional_1/convLSTM_1/Add_1Addfunctional_1/convLSTM_1/Mul:z:0(functional_1/convLSTM_1/Const_3:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/Add_1§
/functional_1/convLSTM_1/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/functional_1/convLSTM_1/clip_by_value/Minimum/y
-functional_1/convLSTM_1/clip_by_value/MinimumMinimum!functional_1/convLSTM_1/Add_1:z:08functional_1/convLSTM_1/clip_by_value/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2/
-functional_1/convLSTM_1/clip_by_value/Minimum
'functional_1/convLSTM_1/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'functional_1/convLSTM_1/clip_by_value/yù
%functional_1/convLSTM_1/clip_by_valueMaximum1functional_1/convLSTM_1/clip_by_value/Minimum:z:00functional_1/convLSTM_1/clip_by_value/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2'
%functional_1/convLSTM_1/clip_by_valueâ
functional_1/convLSTM_1/add_2AddV2.functional_1/convLSTM_1/convolution_2:output:0.functional_1/convLSTM_1/convolution_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/add_2
functional_1/convLSTM_1/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2!
functional_1/convLSTM_1/Const_4
functional_1/convLSTM_1/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_1/Const_5Í
functional_1/convLSTM_1/Mul_1Mul!functional_1/convLSTM_1/add_2:z:0(functional_1/convLSTM_1/Const_4:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/Mul_1Í
functional_1/convLSTM_1/Add_3Add!functional_1/convLSTM_1/Mul_1:z:0(functional_1/convLSTM_1/Const_5:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/Add_3«
1functional_1/convLSTM_1/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1functional_1/convLSTM_1/clip_by_value_1/Minimum/y
/functional_1/convLSTM_1/clip_by_value_1/MinimumMinimum!functional_1/convLSTM_1/Add_3:z:0:functional_1/convLSTM_1/clip_by_value_1/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv21
/functional_1/convLSTM_1/clip_by_value_1/Minimum
)functional_1/convLSTM_1/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)functional_1/convLSTM_1/clip_by_value_1/y
'functional_1/convLSTM_1/clip_by_value_1Maximum3functional_1/convLSTM_1/clip_by_value_1/Minimum:z:02functional_1/convLSTM_1/clip_by_value_1/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2)
'functional_1/convLSTM_1/clip_by_value_1Û
functional_1/convLSTM_1/mul_2Mul+functional_1/convLSTM_1/clip_by_value_1:z:0,functional_1/convLSTM_1/convolution:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/mul_2â
functional_1/convLSTM_1/add_4AddV2.functional_1/convLSTM_1/convolution_3:output:0.functional_1/convLSTM_1/convolution_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/add_4¢
functional_1/convLSTM_1/TanhTanh!functional_1/convLSTM_1/add_4:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/TanhÍ
functional_1/convLSTM_1/mul_3Mul)functional_1/convLSTM_1/clip_by_value:z:0 functional_1/convLSTM_1/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/mul_3È
functional_1/convLSTM_1/add_5AddV2!functional_1/convLSTM_1/mul_2:z:0!functional_1/convLSTM_1/mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/add_5â
functional_1/convLSTM_1/add_6AddV2.functional_1/convLSTM_1/convolution_4:output:0.functional_1/convLSTM_1/convolution_8:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/add_6
functional_1/convLSTM_1/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2!
functional_1/convLSTM_1/Const_6
functional_1/convLSTM_1/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
functional_1/convLSTM_1/Const_7Í
functional_1/convLSTM_1/Mul_4Mul!functional_1/convLSTM_1/add_6:z:0(functional_1/convLSTM_1/Const_6:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/Mul_4Í
functional_1/convLSTM_1/Add_7Add!functional_1/convLSTM_1/Mul_4:z:0(functional_1/convLSTM_1/Const_7:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/Add_7«
1functional_1/convLSTM_1/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1functional_1/convLSTM_1/clip_by_value_2/Minimum/y
/functional_1/convLSTM_1/clip_by_value_2/MinimumMinimum!functional_1/convLSTM_1/Add_7:z:0:functional_1/convLSTM_1/clip_by_value_2/Minimum/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv21
/functional_1/convLSTM_1/clip_by_value_2/Minimum
)functional_1/convLSTM_1/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)functional_1/convLSTM_1/clip_by_value_2/y
'functional_1/convLSTM_1/clip_by_value_2Maximum3functional_1/convLSTM_1/clip_by_value_2/Minimum:z:02functional_1/convLSTM_1/clip_by_value_2/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2)
'functional_1/convLSTM_1/clip_by_value_2¦
functional_1/convLSTM_1/Tanh_1Tanh!functional_1/convLSTM_1/add_5:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2 
functional_1/convLSTM_1/Tanh_1Ñ
functional_1/convLSTM_1/mul_5Mul+functional_1/convLSTM_1/clip_by_value_2:z:0"functional_1/convLSTM_1/Tanh_1:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv2
functional_1/convLSTM_1/mul_5Ç
5functional_1/convLSTM_1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         27
5functional_1/convLSTM_1/TensorArrayV2_1/element_shape
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
functional_1/convLSTM_1/time¯
0functional_1/convLSTM_1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0functional_1/convLSTM_1/while/maximum_iterations
*functional_1/convLSTM_1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2,
*functional_1/convLSTM_1/while/loop_counter
functional_1/convLSTM_1/whileWhile3functional_1/convLSTM_1/while/loop_counter:output:09functional_1/convLSTM_1/while/maximum_iterations:output:0%functional_1/convLSTM_1/time:output:00functional_1/convLSTM_1/TensorArrayV2_1:handle:0,functional_1/convLSTM_1/convolution:output:0,functional_1/convLSTM_1/convolution:output:0.functional_1/convLSTM_1/strided_slice:output:0Ofunctional_1/convLSTM_1/TensorArrayUnstack/TensorListFromTensor:output_handle:05functional_1_convlstm_1_split_readvariableop_resource7functional_1_convlstm_1_split_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*\
_output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *$
_read_only_resource_inputs
	*5
body-R+
)functional_1_convLSTM_1_while_body_576280*5
cond-R+
)functional_1_convLSTM_1_while_cond_576279*[
output_shapesJ
H: : : : :ÿÿÿÿÿÿÿÿÿv:ÿÿÿÿÿÿÿÿÿv: : : : *
parallel_iterations 2
functional_1/convLSTM_1/whileí
Hfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"ÿÿÿÿv         2J
Hfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shapeÑ
:functional_1/convLSTM_1/TensorArrayV2Stack/TensorListStackTensorListStack&functional_1/convLSTM_1/while:output:3Qfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv*
element_dtype02<
:functional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack±
-functional_1/convLSTM_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2/
-functional_1/convLSTM_1/strided_slice_2/stack¬
/functional_1/convLSTM_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/convLSTM_1/strided_slice_2/stack_1¬
/functional_1/convLSTM_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/functional_1/convLSTM_1/strided_slice_2/stack_2³
'functional_1/convLSTM_1/strided_slice_2StridedSliceCfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:06functional_1/convLSTM_1/strided_slice_2/stack:output:08functional_1/convLSTM_1/strided_slice_2/stack_1:output:08functional_1/convLSTM_1/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿv*
shrink_axis_mask2)
'functional_1/convLSTM_1/strided_slice_2±
(functional_1/convLSTM_1/transpose_1/permConst*
_output_shapes
:*
dtype0*)
value B"                2*
(functional_1/convLSTM_1/transpose_1/perm
#functional_1/convLSTM_1/transpose_1	TransposeCfunctional_1/convLSTM_1/TensorArrayV2Stack/TensorListStack:tensor:01functional_1/convLSTM_1/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿv2%
#functional_1/convLSTM_1/transpose_1ò
8functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02:
8functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpø
:functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02<
:functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1è
)functional_1/batchnorm_1/FusedBatchNormV3FusedBatchNormV30functional_1/convLSTM_1/strided_slice_2:output:0functional_1_batchnorm_1_scalefunctional_1_batchnorm_1_offset@functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:ÿÿÿÿÿÿÿÿÿv:::::*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_1/FusedBatchNormV3Ã
7functional_1/global_max_pooling2d/Max/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/global_max_pooling2d/Max/reduction_indicesø
%functional_1/global_max_pooling2d/MaxMax-functional_1/batchnorm_1/FusedBatchNormV3:y:0@functional_1/global_max_pooling2d/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%functional_1/global_max_pooling2d/MaxÆ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp1functional_1_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpÔ
functional_1/dense/MatMulMatMul.functional_1/global_max_pooling2d/Max:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense/MatMulÅ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_1_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpÍ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense/BiasAdd
functional_1/dense/SigmoidSigmoid#functional_1/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense/Sigmoid
IdentityIdentityfunctional_1/dense/Sigmoid:y:0^functional_1/convLSTM_1/while*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿx ::::::::2>
functional_1/convLSTM_1/whilefunctional_1/convLSTM_1/while:] Y
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿx 
!
_user_specified_name	input_1"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*µ
serving_default¡
H
input_1=
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿx 9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ý·
Y
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
regularization_losses
		variables

	keras_api

signatures
g_default_save_signature
h__call__
*i&call_and_return_all_conditional_losses"¨V
_tf_keras_networkV{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ConvLSTM2D", "config": {"name": "convLSTM_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "name": "convLSTM_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_1", "inbound_nodes": [[["convLSTM_1", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "global_max_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling2d", "inbound_nodes": [[["batchnorm_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_max_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 120, 160, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "ConvLSTM2D", "config": {"name": "convLSTM_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "name": "convLSTM_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batchnorm_1", "inbound_nodes": [[["convLSTM_1", 0, 0, {}]]]}, {"class_name": "GlobalMaxPooling2D", "config": {"name": "global_max_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_max_pooling2d", "inbound_nodes": [[["batchnorm_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_max_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy_in_loss", "from_logits": false, "label_smoothing": 0}}, "metrics": [{"class_name": "BinaryCrossentropy", "config": {"name": "binary_crossentropy_in_metrics", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 7.999999797903001e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
"
_tf_keras_input_layerâ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 30, 120, 160, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
í
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"Ä
_tf_keras_rnn_layer¦{"class_name": "ConvLSTM2D", "name": "convLSTM_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "convLSTM_1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 120, 160, 4]}, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 120, 160, 4]}}
	
axis
moving_mean
moving_variance
regularization_losses
trainable_variables
	variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "BatchNormalization", "name": "batchnorm_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batchnorm_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": false, "scale": false, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 118, 158, 16]}}

regularization_losses
trainable_variables
	variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"ø
_tf_keras_layerÞ{"class_name": "GlobalMaxPooling2D", "name": "global_max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_max_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
î

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
p__call__
*q&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}

#iter

$beta_1

%beta_2
	&decay
'learning_ratem_m`(ma)mbvcvd(ve)vf"
	optimizer
<
(0
)1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
J
(0
)1
2
3
4
5"
trackable_list_wrapper
Ê
trainable_variables
regularization_losses
*layer_regularization_losses

+layers
,layer_metrics
-metrics
.non_trainable_variables
		variables
h__call__
g_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
rserving_default"
signature_map
À


(kernel
)recurrent_kernel
/regularization_losses
0trainable_variables
1	variables
2	keras_api
s__call__
*t&call_and_return_all_conditional_losses"	
_tf_keras_layerõ{"class_name": "ConvLSTM2DCell", "name": "conv_lst_m2d_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv_lst_m2d_cell", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0}}, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
¹

3states
trainable_variables
regularization_losses
4layer_regularization_losses

5layers
6layer_metrics
7metrics
8non_trainable_variables
	variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% (2batchnorm_1/moving_mean
+:) (2batchnorm_1/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
9layer_regularization_losses
regularization_losses
trainable_variables

:layers
;layer_metrics
<metrics
=non_trainable_variables
	variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
>layer_regularization_losses
regularization_losses
trainable_variables

?layers
@layer_metrics
Ametrics
Bnon_trainable_variables
	variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Clayer_regularization_losses
regularization_losses
 trainable_variables

Dlayers
Elayer_metrics
Fmetrics
Gnon_trainable_variables
!	variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
+:)@2convLSTM_1/kernel
5:3@2convLSTM_1/recurrent_kernel
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
5
H0
I1
J2"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
u0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
­
Klayer_regularization_losses
/regularization_losses
0trainable_variables

Llayers
Mlayer_metrics
Nmetrics
Onon_trainable_variables
1	variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!"
trackable_tuple_wrapper
 "
trackable_list_wrapper
'
0"
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
0
1"
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
º
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"ó
_tf_keras_metricØ{"class_name": "BinaryCrossentropy", "name": "binary_crossentropy_in_metrics", "dtype": "float32", "config": {"name": "binary_crossentropy_in_metrics", "dtype": "float32", "from_logits": false, "label_smoothing": 0}}
¯"
Ytrue_positives
Ztrue_negatives
[false_positives
\false_negatives
]	variables
^	keras_api"¼!
_tf_keras_metric¡!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
'
u0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
<
Y0
Z1
[2
\3"
trackable_list_wrapper
-
]	variables"
_generic_user_object
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
0:.@2Adam/convLSTM_1/kernel/m
::8@2"Adam/convLSTM_1/recurrent_kernel/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
0:.@2Adam/convLSTM_1/kernel/v
::8@2"Adam/convLSTM_1/recurrent_kernel/v
ì2é
!__inference__wrapped_model_576416Ã
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *3¢0
.+
input_1ÿÿÿÿÿÿÿÿÿx 
2ÿ
-__inference_functional_1_layer_call_fn_577753
-__inference_functional_1_layer_call_fn_578273
-__inference_functional_1_layer_call_fn_577707
-__inference_functional_1_layer_call_fn_578252À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_functional_1_layer_call_and_return_conditional_losses_578231
H__inference_functional_1_layer_call_and_return_conditional_losses_578009
H__inference_functional_1_layer_call_and_return_conditional_losses_577635
H__inference_functional_1_layer_call_and_return_conditional_losses_577660À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_convLSTM_1_layer_call_fn_579105
+__inference_convLSTM_1_layer_call_fn_578694
+__inference_convLSTM_1_layer_call_fn_579114
+__inference_convLSTM_1_layer_call_fn_578685Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
û2ø
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578475
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_579096
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578676
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578895Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
,__inference_batchnorm_1_layer_call_fn_579174
,__inference_batchnorm_1_layer_call_fn_579161
,__inference_batchnorm_1_layer_call_fn_579234
,__inference_batchnorm_1_layer_call_fn_579221´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579208
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579192
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579132
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579148´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
5__inference_global_max_pooling2d_layer_call_fn_577107à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¸2µ
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_577101à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ð2Í
&__inference_dense_layer_call_fn_579254¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dense_layer_call_and_return_conditional_losses_579245¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
3B1
$__inference_signature_wrapper_577785input_1
¬2©
2__inference_conv_lst_m2d_cell_layer_call_fn_579404
2__inference_conv_lst_m2d_cell_layer_call_fn_579419¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_579322
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_579389¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
³2°
__inference_loss_fn_0_579424
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
	J
Const
J	
Const_1
!__inference__wrapped_model_576416x()vw=¢:
3¢0
.+
input_1ÿÿÿÿÿÿÿÿÿx 
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ¿
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579132tvw<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿv
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿv
 ¿
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579148tvw<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿv
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿv
 â
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579192vwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 â
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_579208vwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_batchnorm_1_layer_call_fn_579161gvw<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿv
p
ª "!ÿÿÿÿÿÿÿÿÿv
,__inference_batchnorm_1_layer_call_fn_579174gvw<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿv
p 
ª "!ÿÿÿÿÿÿÿÿÿvº
,__inference_batchnorm_1_layer_call_fn_579221vwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
,__inference_batchnorm_1_layer_call_fn_579234vwM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578475~()H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿx 

 
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿv
 È
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578676~()H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿx 

 
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿv
 Ù
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_578895()X¢U
N¢K
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 

 
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿv
 Ù
F__inference_convLSTM_1_layer_call_and_return_conditional_losses_579096()X¢U
N¢K
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 

 
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿv
  
+__inference_convLSTM_1_layer_call_fn_578685q()H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿx 

 
p

 
ª "!ÿÿÿÿÿÿÿÿÿv 
+__inference_convLSTM_1_layer_call_fn_578694q()H¢E
>¢;
-*
inputsÿÿÿÿÿÿÿÿÿx 

 
p 

 
ª "!ÿÿÿÿÿÿÿÿÿv±
+__inference_convLSTM_1_layer_call_fn_579105()X¢U
N¢K
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 

 
p

 
ª "!ÿÿÿÿÿÿÿÿÿv±
+__inference_convLSTM_1_layer_call_fn_579114()X¢U
N¢K
=:
85
inputs/0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿx 

 
p 

 
ª "!ÿÿÿÿÿÿÿÿÿv
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_579322¹()¢
¢
)&
inputsÿÿÿÿÿÿÿÿÿx 
]¢Z
+(
states/0ÿÿÿÿÿÿÿÿÿv
+(
states/1ÿÿÿÿÿÿÿÿÿv
p
ª "¢
¢
&#
0/0ÿÿÿÿÿÿÿÿÿv
WT
(%
0/1/0ÿÿÿÿÿÿÿÿÿv
(%
0/1/1ÿÿÿÿÿÿÿÿÿv
 
M__inference_conv_lst_m2d_cell_layer_call_and_return_conditional_losses_579389¹()¢
¢
)&
inputsÿÿÿÿÿÿÿÿÿx 
]¢Z
+(
states/0ÿÿÿÿÿÿÿÿÿv
+(
states/1ÿÿÿÿÿÿÿÿÿv
p 
ª "¢
¢
&#
0/0ÿÿÿÿÿÿÿÿÿv
WT
(%
0/1/0ÿÿÿÿÿÿÿÿÿv
(%
0/1/1ÿÿÿÿÿÿÿÿÿv
 Ü
2__inference_conv_lst_m2d_cell_layer_call_fn_579404¥()¢
¢
)&
inputsÿÿÿÿÿÿÿÿÿx 
]¢Z
+(
states/0ÿÿÿÿÿÿÿÿÿv
+(
states/1ÿÿÿÿÿÿÿÿÿv
p
ª "~¢{
$!
0ÿÿÿÿÿÿÿÿÿv
SP
&#
1/0ÿÿÿÿÿÿÿÿÿv
&#
1/1ÿÿÿÿÿÿÿÿÿvÜ
2__inference_conv_lst_m2d_cell_layer_call_fn_579419¥()¢
¢
)&
inputsÿÿÿÿÿÿÿÿÿx 
]¢Z
+(
states/0ÿÿÿÿÿÿÿÿÿv
+(
states/1ÿÿÿÿÿÿÿÿÿv
p 
ª "~¢{
$!
0ÿÿÿÿÿÿÿÿÿv
SP
&#
1/0ÿÿÿÿÿÿÿÿÿv
&#
1/1ÿÿÿÿÿÿÿÿÿv¡
A__inference_dense_layer_call_and_return_conditional_losses_579245\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
&__inference_dense_layer_call_fn_579254O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÄ
H__inference_functional_1_layer_call_and_return_conditional_losses_577635x()vwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿx 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
H__inference_functional_1_layer_call_and_return_conditional_losses_577660x()vwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿx 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
H__inference_functional_1_layer_call_and_return_conditional_losses_578009w()vwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿx 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
H__inference_functional_1_layer_call_and_return_conditional_losses_578231w()vwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿx 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_functional_1_layer_call_fn_577707k()vwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿx 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_functional_1_layer_call_fn_577753k()vwE¢B
;¢8
.+
input_1ÿÿÿÿÿÿÿÿÿx 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_functional_1_layer_call_fn_578252j()vwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿx 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_functional_1_layer_call_fn_578273j()vwD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿx 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÙ
P__inference_global_max_pooling2d_layer_call_and_return_conditional_losses_577101R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 °
5__inference_global_max_pooling2d_layer_call_fn_577107wR¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
__inference_loss_fn_0_579424¢

¢ 
ª " ¬
$__inference_signature_wrapper_577785()vwH¢E
¢ 
>ª;
9
input_1.+
input_1ÿÿÿÿÿÿÿÿÿx "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ