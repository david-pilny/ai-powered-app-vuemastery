<script setup>
import Transcriber from '../components/Transcriber.vue'
import Uploader from '../components/Uploader.vue'
</script>

<template>
  <article class="grid grid-cols-1 lg:grid-cols-2 gap-x-8">
    <div class="mt-4">
      <h3>Choose audio</h3>
      <!-- Uploader -->
      <section class="my-4">
        <Uploader />
      </section>
      <!-- Transcriber -->
      <Transcriber />
    </div>
    <div class="min-h-80 flex flex-col justify-between">
      <!-- Chat Window -->
    </div>
  </article>
</template>
