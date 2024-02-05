# Design Document: fgbuster JAX Integration

- [1. Overview](#1-overview)
  - [1.1. Current Infrastructure](#11-current-infrastructure)
- [2. Proposed Changes](#2-proposed-changes)
  - [2.1. Transform inputs and analytic component arg to Pytrees **❌**](#21-transform-inputs-and-analytic-component-arg-to-pytrees-)
  - [2.2. Transition to JAX  **✅**](#22-transition-to-jax--)
  - [2.3. Ensure evaluation and diff evaluation is good on mixing matrix \[ \]](#23-ensure-evaluation-and-diff-evaluation-is-good-on-mixing-matrix--)
  - [2.4. Einsum is not working with jax.numpy.einsum](#24-einsum-is-not-working-with-jaxnumpyeinsum)
  - [2.5. Replace minimiser with optax.adam for testing](#25-replace-minimiser-with-optaxadam-for-testing)
  - [2.6. Integrate Lineax for the matrix operation part](#26-integrate-lineax-for-the-matrix-operation-part)

## 1. Overview

The primary objective of this project is to enhance the infrastructure of fgbuster, a component separation library for Cosmic Microwave Background (CMB) data analysis, by replacing the existing likelihood minimization with a JAX-based optimizer. This transition to JAX aims to leverage GPU capabilities for more efficient calculations, specifically utilizing the optax library, with a focus on the Adam optimizer.

### 1.1. Current Infrastructure

The core of fgbuster relies on the `AnalyticComponent` class, which employs Sympy for parsing mathematical expressions and symbolic differentiation. The optimization process currently utilizes Scipy's `minimize` function to find optimal parameter values for the model.

## 2. Proposed Changes

### 2.1. Transform inputs and analytic component arg to Pytrees **&#10060;**

This would make jit eval and grad easy, but it would make super difficult to chose argnum gradient target\
We never differentate W.R.T nu and sometimes we wan't to ignore fixed parameters

### 2.2. Transition to JAX  **&#9989;**

Component's eval is now jittable but there is a caveat, the auto batching done by sympy gives weird gradient shaptes

Solution is to write custom broadcasting function `make_broadcastable` 

Grad is now identical in output to that of numpy , but orders of magnitude faster


### 2.3. Ensure evaluation and diff evaluation is good on mixing matrix [ ]

On going


### 2.4. Einsum is not working with jax.numpy.einsum

To be checked

A possible solution could be to replace it with matmul and addition\
Or even CuTensor from nvidia (rocmTensor from AMD)

### 2.5. Replace minimiser with optax.adam for testing


### 2.6. Integrate Lineax for the matrix operation part